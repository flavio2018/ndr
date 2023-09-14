import torch


def cross_entropy_2d(outputs, target, targets_lengths):
	cross_entropy_no_reduction = torch.nn.CrossEntropyLoss(reduction="none")
	cumulative_loss = 0
	pos_mask = torch.ones_like(target).cumsum(1)
	mask = pos_mask <= targets_lengths.unsqueeze(1)
	for char_pos in range(target.size(1)):
		char_loss = cross_entropy_no_reduction(outputs[:, char_pos, :], target[:, char_pos])
		masked_char_loss = char_loss * mask[:, char_pos]
		cumulative_loss += masked_char_loss.sum()
	avg_loss = cumulative_loss / mask.sum()
	return avg_loss
