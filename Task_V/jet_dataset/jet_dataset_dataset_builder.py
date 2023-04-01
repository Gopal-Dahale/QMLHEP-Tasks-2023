"""jet_dataset dataset."""

import tensorflow_datasets as tfds
import tensorflow as tf
from pathlib import Path
import numpy as np
from tqdm import tqdm
import h5py

class Builder(tfds.core.GeneratorBasedBuilder):
	"""DatasetBuilder for jet_dataset dataset."""

	VERSION = tfds.core.Version('1.0.0')
	RELEASE_NOTES = {
		'1.0.0': 'Initial release.',
	}

	def _info(self) -> tfds.core.DatasetInfo:
		"""Returns the dataset metadata."""
		# TODO(my_dataset): Specifies the tfds.core.DatasetInfo object
		return self.dataset_info_from_configs(
				homepage='https://dataset-homepage.org',
				features=tfds.features.FeaturesDict({
						# These are the features of your dataset like images, labels ...
						'num_nodes': tfds.features.Tensor(shape=(), dtype=tf.int64),
						'num_edges': tfds.features.Tensor(shape=(), dtype=tf.int64),
						'node_feats': tfds.features.Tensor(shape=(None, 4), dtype=tf.float64),
						'edge_feats': tfds.features.Tensor(shape=(None, 1), dtype=tf.float64),
						'edge_index': tfds.features.Tensor(shape=(2, None), dtype=tf.int64),
						'label': tfds.features.ClassLabel(num_classes=2), # Here, 'label' can be 0 or 1.
				}),
				# If there's a common (input, target) tuple from the
				# features, specify them here. They'll be used if
				# `as_supervised=True` in `builder.as_dataset`.
				supervised_keys=None,  # Set to `None` to disable
				# Specify whether to disable shuffling on the examples. Set to False by default.
				disable_shuffling=False,
			)

	def _split_generators(self, dl_manager: tfds.download.DownloadManager):
		"""Returns SplitGenerators."""
		# TODO(jet_dataset): Downloads the data and defines the splits
		path = Path(__file__).parents[2]/'data'/'raw'/ 'QG_jets_split.h5'
		
		# TODO(my_dataset): Returns the Dict[split names, Iterator[Key, Example]]
		return {
			'train': self._generate_examples(path, 'train'),
			'val': self._generate_examples(path, 'val'),
			'test': self._generate_examples(path, 'test'),
		}

	def _generate_examples(self, path, split):
		"""Yields examples."""
		# TODO(my_dataset): Yields (key, example) tuples from the dataset
		PID2FLOAT_MAP = {22: 0,
						211: .1, -211: .2,
						321: .3, -321: .4,
						130: .5,
						2112: .6, -2112: .7,
						2212: .8, -2212: .9,
						11: 1.0, -11: 1.1,
						13: 1.2, -13: 1.3,
						0: 0,}

		with h5py.File(path, "r") as f:
			X= f[f"x_{split}"][:]
			y = f[f"y_{split}"][:].squeeze().astype(int)

		n_graphs = X.shape[0]
		pids = np.unique(X[:, :, 3].flatten())
		for pid in tqdm(pids):
			np.place(X[:, :, 3], X[:, :, 3] == pid, PID2FLOAT_MAP[pid])
				
		for i in tqdm(range(n_graphs)):
			jet_data = X[i]
			jet_label = y[i]
			
			_jet_data = jet_data[~np.all(jet_data == 0, axis = 1)] # Remove zero padded entries
	
			# Centering jets and normalizing pT
			yphi_avg = np.average(_jet_data[:,1:3], weights=_jet_data[:,0], axis=0)
			_jet_data[:,1:3] -= yphi_avg
			_jet_data[:, 0] /= np.sum(_jet_data[:, 0])

			# Sort by pT (0th column)
			_jet_data = _jet_data[_jet_data[:,0].argsort()][::-1].copy()
			
			# Get node features
			node_feats = self._get_node_features(_jet_data)
			
			# Get adjacency info
			edge_index = self._get_adjacency_info(_jet_data)
			
			# Get labels info
			label = self._get_labels(jet_label)

			# And yield (key, feature_dict)
			yield i, {
				'num_nodes': node_feats.shape[0],
				'num_edges': edge_index.shape[1],
				'node_feats': node_feats,
				'edge_feats': np.ones((edge_index.shape[1], 1)),
				'edge_index': edge_index,
				'label': label,
			}
	
	def _get_node_features(self, jet_data):
		""" 
		This will return a matrix / 2d array of the shape
		[Number of Nodes, Node Feature size]
		"""
		
		return jet_data

	def _get_adjacency_info(self, jet_data):
		"""
		We could also use rdmolops.GetAdjacencyMatrix(mol)
		but we want to be sure that the order of the indices
		matches the order of the edge features
		"""
		
		pt_order = jet_data[:,0].argsort()[::-1]
		rapidity_order = jet_data[:,1].argsort()[::-1]
		eta_order = jet_data[:,2].argsort()[::-1]
		
		in_node  = np.concatenate((pt_order[:-1], rapidity_order[:-1], eta_order[:-1]))
		out_node = np.concatenate((pt_order[1: ], rapidity_order[1: ], eta_order[1: ]))
		
		edge_indices = np.stack((in_node, out_node), axis= 0)
		return edge_indices
	   
	def _get_labels(self, label):
		return label
