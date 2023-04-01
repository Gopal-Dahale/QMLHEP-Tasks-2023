import input_pipeline
import jax
from utils import replace_globals, create_optimizer, add_prefix_to_keys
import flax.linen as nn
from clu import parameter_overview
import models
from flax.training import train_state
import numpy as np
import jraph
from typing import Dict, Iterable, Tuple, Optional
from clu import metric_writers
from clu import metrics
from clu import parameter_overview
from clu import periodic_actions
import jax.numpy as jnp
from metrics import TrainMetrics, EvalMetrics
import tensorflow as tf
from absl import logging
from clu import platform
from absl import app
from absl import flags
from ml_collections import config_flags

FLAGS = flags.FLAGS
flags.DEFINE_string('workdir', None, 'Directory to store model data.')
config_flags.DEFINE_config_file(
    'config',
    None,
    'File path to the training hyperparameter configuration.',
    lock_config=True)

def create_model(config,
                 deterministic: bool) -> nn.Module:
  """Creates a Flax model, as specified by the config."""
  if config.model == 'GraphNet':
    return models.GraphNet(
        latent_size=config.latent_size,
        num_mlp_layers=config.num_mlp_layers,
        message_passing_steps=config.message_passing_steps,
        output_globals_size=config.num_classes,
        dropout_rate=config.dropout_rate,
        skip_connections=config.skip_connections,
        layer_norm=config.layer_norm,
        use_edge_model=config.use_edge_model,
        deterministic=deterministic)
  if config.model == 'GraphConvNet':
    return models.GraphConvNet(
        latent_size=config.latent_size,
        num_mlp_layers=config.num_mlp_layers,
        message_passing_steps=config.message_passing_steps,
        output_globals_size=config.num_classes,
        dropout_rate=config.dropout_rate,
        skip_connections=config.skip_connections,
        layer_norm=config.layer_norm,
        deterministic=deterministic)
  raise ValueError(f'Unsupported model: {config.model}.')

def get_valid_mask(labels: jnp.ndarray,
                   graphs: jraph.GraphsTuple) -> jnp.ndarray:
  """Gets the binary mask indicating only valid labels and graphs."""
  # We have to ignore all NaN values - which indicate labels for which
  # the current graphs have no label.
  labels_mask = ~jnp.isnan(labels)

  # Since we have extra 'dummy' graphs in our batch due to padding, we want
  # to mask out any loss associated with the dummy graphs.
  # Since we padded with `pad_with_graphs` we can recover the mask by using
  # get_graph_padding_mask.
  graph_mask = jraph.get_graph_padding_mask(graphs)

  # Combine the mask over labels with the mask over graphs.
  return labels_mask & graph_mask[:, None]


def get_predicted_logits(state: train_state.TrainState,
                         graphs: jraph.GraphsTuple,
                         rngs: Optional[Dict[str, jnp.ndarray]]) -> jnp.ndarray:
    """Get predicted logits from the network for input graphs."""
    pred_graphs = state.apply_fn(state.params, graphs, rngs=rngs)
    logits = pred_graphs.globals
    return logits

def binary_cross_entropy_with_mask(*, logits: jnp.ndarray, labels: jnp.ndarray,
                                   mask: jnp.ndarray):
  """Binary cross entropy loss for unnormalized logits, with masked elements."""

  assert logits.shape == labels.shape == mask.shape
  assert len(logits.shape) == 2

  # To prevent propagation of NaNs during grad().
  # We mask over the loss for invalid targets later.
  labels = jnp.where(mask, labels, -1)

  # Numerically stable implementation of BCE loss.
  # This mimics TensorFlow's tf.nn.sigmoid_cross_entropy_with_logits().
  positive_logits = (logits >= 0)
  relu_logits = jnp.where(positive_logits, logits, 0)
  abs_logits = jnp.where(positive_logits, logits, -logits)
  return relu_logits - (logits * labels) + (
      jnp.log(1 + jnp.exp(-abs_logits)))

@jax.jit
def train_step(
    state: train_state.TrainState, graphs: jraph.GraphsTuple,
    rngs: Dict[str, jnp.ndarray]
) -> Tuple[train_state.TrainState, metrics.Collection]:
  """Performs one update step over the current batch of graphs."""

  def loss_fn(params, graphs):
    curr_state = state.replace(params=params)

    # Extract labels.
    labels = graphs.globals
    labels = jax.nn.one_hot(labels, 2)

    # Replace the global feature for graph classification.
    graphs = replace_globals(graphs)

    # Compute logits and resulting loss.
    logits = get_predicted_logits(curr_state, graphs, rngs)
    mask = get_valid_mask(labels, graphs)
    loss = binary_cross_entropy_with_mask(
        logits=logits, labels=labels, mask=mask)
    mean_loss = jnp.sum(jnp.where(mask, loss, 0)) / jnp.sum(mask)

    return mean_loss, (loss, logits, labels, mask)

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (_, (loss, logits, labels, mask)), grads = grad_fn(state.params, graphs)
  state = state.apply_gradients(grads=grads)

  metrics_update = TrainMetrics.single_from_model_output(
      loss=loss, logits=logits, labels=labels, mask=mask)
  return state, metrics_update

@jax.jit
def evaluate_step(
    state: train_state.TrainState,
    graphs: jraph.GraphsTuple,
) -> metrics.Collection:
    """Computes metrics over a set of graphs."""

    # The target labels our model has to predict.
    labels = graphs.globals
    labels = jax.nn.one_hot(labels, 2)

    # Replace the global feature for graph classification.
    graphs = replace_globals(graphs)

    # Get predicted logits, and corresponding probabilities.
    logits = get_predicted_logits(state, graphs, rngs=None)

    # Get the mask for valid labels and graphs.
    mask = get_valid_mask(labels, graphs)

    # Compute the various metrics.
    loss = binary_cross_entropy_with_mask(logits=logits, labels=labels, mask=mask)

    return EvalMetrics.single_from_model_output(
        loss=loss, logits=logits, labels=labels, mask=mask)

def evaluate_model(state: train_state.TrainState,
                   datasets: Dict[str, tf.data.Dataset],
                   splits: Iterable[str]) -> Dict[str, metrics.Collection]:
    """Evaluates the model on metrics over the specified splits."""

    # Loop over each split independently.
    eval_metrics = {}
    for split in splits:
        split_metrics = None

        # Loop over graphs.
        for graphs in datasets[split].as_numpy_iterator():
            split_metrics_update = evaluate_step(state, graphs)
            
            # Update metrics.
            if split_metrics is None:
                split_metrics = split_metrics_update
            else:
                split_metrics = split_metrics.merge(split_metrics_update)
        eval_metrics[split] = split_metrics
    return eval_metrics  # pytype: disable=bad-return-type


def train_and_evaluate(config, workdir):

    # We only support single-host training.
    assert jax.process_count() == 1

    # Create writer for logs.
    writer = metric_writers.create_default_writer(workdir)
    writer.write_hparams(config.to_dict())

    # Get datasets, organized by split.
    logging.info('Obtaining datasets.')
    datasets = input_pipeline.get_datasets(
        config.batch_size,
        add_virtual_node=config.add_virtual_node,
        add_undirected_edges=config.add_undirected_edges,
        add_self_loops=config.add_self_loops)
    train_iter = iter(datasets['train'])

    # Create and initialize the network.
    logging.info('Initializing network.')
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)
    init_graphs = next(datasets['train'].as_numpy_iterator())
    init_graphs = replace_globals(init_graphs)
    init_net = create_model(config, deterministic=True)
    params = jax.jit(init_net.init)(init_rng, init_graphs)
    parameter_overview.log_parameter_overview(params)

    # Create the optimizer.
    tx = create_optimizer(config)

    # Create the training state.
    net = create_model(config, deterministic=False)
    state = train_state.TrainState.create(
        apply_fn=net.apply, params=params, tx=tx)

    initial_step = int(state.step) + 1

    # Create the evaluation state, corresponding to a deterministic model.
    eval_net = create_model(config, deterministic=True)
    eval_state = state.replace(apply_fn=eval_net.apply)

    # Hooks called periodically during training.
    report_progress = periodic_actions.ReportProgress(
        num_train_steps=config.num_train_steps, writer=writer)
    profiler = periodic_actions.Profile(num_profile_steps=5, logdir=workdir)
    hooks = [report_progress, profiler]

    # Begin training loop.
    logging.info('Starting training.')
    train_metrics = None
    for step in range(initial_step, config.num_train_steps + 1):

        # Split PRNG key, to ensure different 'randomness' for every step.
        rng, dropout_rng = jax.random.split(rng)

        # Perform one step of training.
        with jax.profiler.StepTraceAnnotation('train', step_num=step):
            graphs = jax.tree_util.tree_map(np.asarray, next(train_iter))
            state, metrics_update = train_step(
                state, graphs, rngs={'dropout': dropout_rng})

            # Update metrics.
            if train_metrics is None:
                train_metrics = metrics_update
            else:
                train_metrics = train_metrics.merge(metrics_update)

        # Quick indication that training is happening.
        logging.log_first_n(logging.INFO, 'Finished training step %d.', 10, step)
        for hook in hooks:
            hook(step)
        
        # Log, if required.
        is_last_step = (step == config.num_train_steps - 1)
        if step % config.log_every_steps == 0 or is_last_step:
            writer.write_scalars(step, add_prefix_to_keys(train_metrics.compute(), 'train'))
            train_metrics = None

        # Evaluate on validation and test splits, if required.
        if step % config.eval_every_steps == 0 or is_last_step:
            eval_state = eval_state.replace(params=state.params)

            splits = ['val']
            with report_progress.timed('eval'):
                eval_metrics = evaluate_model(eval_state, datasets, splits=splits)
            for split in splits:
                writer.write_scalars(
                    step, add_prefix_to_keys(eval_metrics[split].compute(), split))

    # Test model
    eval_state = eval_state.replace(params=state.params)
    splits = ['test']
    with report_progress.timed('eval'):
        eval_metrics = evaluate_model(eval_state, datasets, splits=splits)
    for split in splits:
        writer.write_scalars(
            step, add_prefix_to_keys(eval_metrics[split].compute(), split))
    
    return state

def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')
    config = FLAGS.config
    workdir = './logs'

    logging.info(f"Model {config.model}")

    # Hide any GPUs from TensorFlow. Otherwise TF might reserve memory and make
    # it unavailable to JAX.
    tf.config.experimental.set_visible_devices([], 'GPU')

    # This example only supports single-host training on a single device.
    logging.info('JAX host: %d / %d', jax.process_index(), jax.process_count())
    logging.info('JAX local devices: %r', jax.local_devices())

    # Add a note so that we can tell which task is which JAX host.
    # (Depending on the platform task 0 is not guaranteed to be host 0)
    platform.work_unit().set_task_status(f'process_index: {jax.process_index()}, '
                                        f'process_count: {jax.process_count()}')
    platform.work_unit().create_artifact(platform.ArtifactType.DIRECTORY, workdir, 'workdir')

    state = train_and_evaluate(config, workdir)

if __name__ == "__main__":
    app.run(main)