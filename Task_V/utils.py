import jax.numpy as jnp
import jraph
import optax
from typing import Dict, Any
import pennylane as qml

def replace_globals(graphs: jraph.GraphsTuple) -> jraph.GraphsTuple:
	"""Replaces the globals attribute with a constant feature for each graph."""
	return graphs._replace(
			globals=jnp.ones([graphs.n_node.shape[0], 1]))

def create_optimizer(
		config) -> optax.GradientTransformation:
	"""Creates an optimizer, as specified by the config."""
	if config.optimizer == 'adam':
		return optax.adam(
				learning_rate=config.learning_rate)
	if config.optimizer == 'sgd':
		return optax.sgd(
				learning_rate=config.learning_rate,
				momentum=config.momentum)
	raise ValueError(f'Unsupported optimizer: {config.optimizer}.')

def add_prefix_to_keys(result: Dict[str, Any], prefix: str) -> Dict[str, Any]:
  """Adds a prefix to the keys of a dict, returning a new dict."""
  return {f'{prefix}_{key}': val for key, val in result.items()}

def circuit(x, params, n):
	w, b = params
	z = jnp.dot(x, w) + b
	for i in range(n):
		qml.RX(z[i], wires=i)
