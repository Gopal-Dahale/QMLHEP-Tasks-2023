from clu import metrics
import flax
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
import jax.numpy as jnp
import jax

def predictions_match_labels(*, logits: jnp.ndarray, labels: jnp.ndarray,
							 **kwargs) -> jnp.ndarray:
	"""Returns a binary array indicating where predictions match the labels."""
	del kwargs  # Unused.
	preds = (logits > 0)
	return (preds == labels).astype(jnp.float32)

@flax.struct.dataclass
class MeanAveragePrecision(
	metrics.CollectingMetric.from_outputs(('labels', 'logits', 'mask'))):
	"""Computes the mean average precision (mAP) over different tasks."""

	def compute(self):
		# Matches the official OGB evaluation scheme for mean average precision.
		values = super().compute()
		labels = values['labels']
		logits = values['logits']
		mask = values['mask']

		assert logits.shape == labels.shape == mask.shape
		assert len(logits.shape) == 2

		probs = jax.nn.sigmoid(logits)
		num_tasks = labels.shape[1]
		average_precisions = np.full(num_tasks, np.nan)

		for task in range(num_tasks):
			# AP is only defined when there is at least one negative data
			# and at least one positive data.
			is_labeled = mask[:, task]
			if len(np.unique(labels[is_labeled, task])) >= 2:
				average_precisions[task] = average_precision_score(
					labels[is_labeled, task], probs[is_labeled, task])

		# When all APs are NaNs, return NaN. This avoids raising a RuntimeWarning.
		if np.isnan(average_precisions).all():
			return np.nan
		
		return np.nanmean(average_precisions)

@flax.struct.dataclass
class AUC(
	metrics.CollectingMetric.from_outputs(('labels', 'logits', 'mask'))):
	"""Computes the ROC AUC Score"""
	
	def compute(self):
		values = super().compute()
		labels = values['labels']
		logits = values['logits']
		mask = values['mask']

		assert logits.shape == labels.shape == mask.shape
		assert len(logits.shape) == 2

		# take first column of mask
		mask = mask[:, 0]

		# We mask over the AUC score for invalid targets.
		labels = jnp.argmax(labels, axis=-1)
		masked_labels = labels[mask]
		probs = jax.nn.softmax(logits)
		probs = probs[:, 1]
		masked_probs = probs[mask]

		return roc_auc_score(masked_labels, masked_probs)


@flax.struct.dataclass
class EvalMetrics(metrics.Collection):

	accuracy: metrics.Average.from_fun(predictions_match_labels)
	loss: metrics.Average.from_output('loss')
	mean_average_precision: MeanAveragePrecision
	auc: AUC


@flax.struct.dataclass
class TrainMetrics(metrics.Collection):

	accuracy: metrics.Average.from_fun(predictions_match_labels)
	loss: metrics.Average.from_output('loss')