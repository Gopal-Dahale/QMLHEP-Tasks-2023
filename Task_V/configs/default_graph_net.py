"""Defines the default hyperparameters and training configuration.

Uses a GraphNetwork model (https://arxiv.org/abs/1806.01261).
"""

import ml_collections


def get_config():
  """Get the hyperparameter configuration for the GraphNetwork model."""
  config = ml_collections.ConfigDict()

  # Optimizer.
  config.optimizer = 'adam'
  config.learning_rate = 1e-3

  # Training hyperparameters.
  config.batch_size = 256
  config.num_train_steps = 5_000
  config.log_every_steps = 100
  config.eval_every_steps = 100
  config.checkpoint_every_steps = 10_000
  config.add_virtual_node = True
  config.add_undirected_edges = True
  config.add_self_loops = True

  # GNN hyperparameters.
  config.model = 'GraphNet'
  config.message_passing_steps = 5
  config.latent_size = 4
  config.dropout_rate = 0.1
  config.num_mlp_layers = 1
  config.num_classes = 2
  config.use_edge_model = False
  config.skip_connections = True
  config.layer_norm = True
  return config
