import torch


def cross_entropy_2d(outputs, target, targets_lengths):
	cross_entropy_no_reduction = torch.nn.CrossEntropyLoss(reduction="none")
	pos_mask = torch.ones_like(target).cumsum(1)
	mask = pos_mask <= targets_lengths.unsqueeze(1)
	outputs = outputs.permute(0, 2, 1)
	if target.shape[1] < outputs.shape[2]:
		outputs = outputs[:, :, :target.shape[1]]
	batch_loss = cross_entropy_no_reduction(outputs, target)
	masked_batch_loss = batch_loss * mask
	cumulative_loss = masked_batch_loss.sum()
	avg_loss = cumulative_loss / mask.sum()
	return avg_loss
