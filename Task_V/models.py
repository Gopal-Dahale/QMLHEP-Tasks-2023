from typing import Callable, Sequence

from flax import linen as nn
import jax.numpy as jnp
import jraph
import pennylane as qml
import jax
from functools import partial
from utils import circuit

def add_graphs_tuples(graphs: jraph.GraphsTuple,
											other_graphs: jraph.GraphsTuple) -> jraph.GraphsTuple:
	"""Adds the nodes, edges and global features from other_graphs to graphs."""
	return graphs._replace(
			nodes=graphs.nodes + other_graphs.nodes,
			edges=graphs.edges + other_graphs.edges,
			globals=graphs.globals + other_graphs.globals)


dev_node = qml.device("default.qubit", wires=4)
dev_global = qml.device("default.qubit", wires=1)

@jax.jit
@qml.qnode(dev_node)
def node_embedder(x, params, n = 4):
	circuit(x, params, n)
	return [qml.expval(qml.PauliZ(i)) for i in range(n)]
	
@jax.jit
@qml.qnode(dev_global)
def global_embedder(x, params, n = 1):
	circuit(x, params, n)
	return [qml.expval(qml.PauliZ(i)) for i in range(n)]

@jax.jit
@qml.qnode(dev_node)
def qnn(x, params, n = 4):
	circuit(x, params, n)
	return [qml.expval(qml.PauliZ(i)) for i in range(n)]

class HQNN(nn.Module):
	"""A multi-layer perceptron."""

	feature_sizes: Sequence[int]
	dropout_rate: float = 0
	deterministic: bool = True
	activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.tanh
	params_init: Callable = nn.initializers.normal()

	@nn.compact
	def __call__(self, inputs):

		params = self.param('theta', self.params_init, (len(self.feature_sizes), 2, 4))

		x = inputs
		for index, size in enumerate(self.feature_sizes):
			qnn_layer = jax.vmap(partial(qnn, params=params[index]), in_axes=(0))
			x = nn.Dense(features=size)(x)
			x = self.activation(x)
			x = qnn_layer(x)
			x = nn.Dropout(
					rate=self.dropout_rate, deterministic=self.deterministic)(x)
			
		return x

class GraphNet(nn.Module):
	"""A complete Graph Network model defined with Jraph."""

	latent_size: int
	num_mlp_layers: int
	message_passing_steps: int
	output_globals_size: int
	dropout_rate: float = 0
	skip_connections: bool = True
	use_edge_model: bool = True
	layer_norm: bool = True
	deterministic: bool = True
	params_init: Callable = nn.initializers.normal()

	@nn.compact
	def __call__(self, graphs: jraph.GraphsTuple) -> jraph.GraphsTuple:
		
		node_embed_params = self.param('np', self.params_init, (2, 4))
		global_embed_params = self.param('gp', self.params_init, (2, 1))

		node_embedder_fn = partial(node_embedder, params=node_embed_params)
		global_embedder_fn = partial(global_embedder, params=global_embed_params)

		qembed_node_fn = jax.vmap(node_embedder_fn, in_axes=(0))
		qembed_global_fn = jax.vmap(global_embedder_fn, in_axes=(0))

		# We will first linearly project the original features as 'embeddings'.
		embedder = jraph.GraphMapFeatures(
				embed_node_fn=qembed_node_fn,
				embed_edge_fn=None,
				embed_global_fn=qembed_global_fn)

		processed_graphs = embedder(graphs)

		# Now, we will apply a Graph Network once for each message-passing round.
		mlp_feature_sizes = [self.latent_size] * self.num_mlp_layers
		for _ in range(self.message_passing_steps):
			
			if self.use_edge_model:
				update_edge_fn = jraph.concatenated_args(
						HQNN(mlp_feature_sizes,
								dropout_rate=self.dropout_rate,
								deterministic=self.deterministic))
			else:
				update_edge_fn = None

			update_node_fn = jraph.concatenated_args(HQNN(mlp_feature_sizes,
              dropout_rate=self.dropout_rate,
              deterministic=self.deterministic))
			update_global_fn = jraph.concatenated_args(HQNN(mlp_feature_sizes,
              dropout_rate=self.dropout_rate,
              deterministic=self.deterministic))

			graph_net = jraph.GraphNetwork(
					update_node_fn=update_node_fn,
					update_edge_fn=update_edge_fn,
					update_global_fn=update_global_fn)

			if self.skip_connections:
				processed_graphs = add_graphs_tuples(
						graph_net(processed_graphs), processed_graphs)
			else:
				processed_graphs = graph_net(processed_graphs)

			if self.layer_norm:
				processed_graphs = processed_graphs._replace(
						nodes=nn.LayerNorm()(processed_graphs.nodes),
						edges=nn.LayerNorm()(processed_graphs.edges),
						globals=nn.LayerNorm()(processed_graphs.globals),
				)

		# Since our graph-level predictions will be at globals, we will
		# decode to get the required output logits.
		decoder = jraph.GraphMapFeatures(
				embed_global_fn=nn.Dense(self.output_globals_size))
		processed_graphs = decoder(processed_graphs)

		return processed_graphs


class GraphConvNet(nn.Module):
	"""A Graph Convolution Network + Pooling model defined with Jraph."""

	latent_size: int
	num_mlp_layers: int
	message_passing_steps: int
	output_globals_size: int
	dropout_rate: float = 0
	skip_connections: bool = True
	layer_norm: bool = True
	deterministic: bool = True
	params_init: Callable = nn.initializers.normal()
	pooling_fn: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray],  # pytype: disable=annotation-type-mismatch  # jax-ndarray
											 jnp.ndarray] = jraph.segment_mean

	def pool(self, graphs: jraph.GraphsTuple) -> jraph.GraphsTuple:
		"""Pooling operation, taken from Jraph."""

		# Equivalent to jnp.sum(n_node), but JIT-able.
		sum_n_node = graphs.nodes.shape[0]  # pytype: disable=attribute-error  # jax-ndarray
		# To aggregate nodes from each graph to global features,
		# we first construct tensors that map the node to the corresponding graph.
		# Example: if you have `n_node=[1,2]`, we construct the tensor [0, 1, 1].
		n_graph = graphs.n_node.shape[0]
		node_graph_indices = jnp.repeat(
				jnp.arange(n_graph),
				graphs.n_node,
				axis=0,
				total_repeat_length=sum_n_node)
		# We use the aggregation function to pool the nodes per graph.
		pooled = self.pooling_fn(graphs.nodes, node_graph_indices, n_graph)  # pytype: disable=wrong-arg-types  # jax-ndarray
		return graphs._replace(globals=pooled)

	@nn.compact
	def __call__(self, graphs: jraph.GraphsTuple) -> jraph.GraphsTuple:

		node_embed_params = self.param('np', self.params_init, (2, 4))
		node_embedder_fn = partial(node_embedder, params=node_embed_params)
		qembed_node_fn = jax.vmap(node_embedder_fn, in_axes=(0))

		# We will first linearly project the original node features as 'embeddings'.
		embedder = jraph.GraphMapFeatures(embed_node_fn=qembed_node_fn)
		processed_graphs = embedder(graphs)

		# Now, we will apply the GCN once for each message-passing round.
		for _ in range(self.message_passing_steps):
			mlp_feature_sizes = [self.latent_size] * self.num_mlp_layers
			update_node_fn = jraph.concatenated_args(
					HQNN(mlp_feature_sizes,
							dropout_rate=self.dropout_rate,
							deterministic=self.deterministic))
			graph_conv = jraph.GraphConvolution(
					update_node_fn=update_node_fn, add_self_edges=True)

			if self.skip_connections:
				processed_graphs = add_graphs_tuples(
						graph_conv(processed_graphs), processed_graphs)
			else:
				processed_graphs = graph_conv(processed_graphs)

			if self.layer_norm:
				processed_graphs = processed_graphs._replace(
						nodes=nn.LayerNorm()(processed_graphs.nodes),
				)
			

		# We apply the pooling operation to get a 'global' embedding.
		processed_graphs = self.pool(processed_graphs)
		
		# Now, we decode this to get the required output logits.
		decoder = jraph.GraphMapFeatures(
				embed_global_fn=nn.Dense(self.output_globals_size))
		processed_graphs = decoder(processed_graphs)
	
		return processed_graphs
