import torch


def cross_entropy_2d(outputs, target, outputs_lengths):
	cross_entropy_no_reduction = torch.nn.CrossEntropyLoss(reduction="none")
	if not isinstance(outputs, list):
		outputs = [outputs[:, pos, :] for pos in range(outputs.size(1))]
	cumulative_loss = 0
	pos_mask = torch.ones_like(target).cumsum(1)  # sum on cols not on rows
	mask = pos_mask <= outputs_lengths.unsqueeze(1)
	for char_pos, output in enumerate(outputs):
		char_loss = cross_entropy_no_reduction(output, target[:, char_pos].squeeze())
		masked_char_loss = char_loss * mask[:, char_pos]
		cumulative_loss += masked_char_loss.sum()
	avg_loss = cumulative_loss / mask.sum()
	return avg_loss
