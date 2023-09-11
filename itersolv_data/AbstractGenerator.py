import torch
from collections import OrderedDict
import torch.nn.functional as F
from torchtext.vocab import vocab
from torchtext.transforms import ToTensor, VocabTransform


_EOS = '.'
_SOS = '?'
_PAD = '#'
_SEP = '/'
_HAL = '$'


class AbstractGenerator:

	def __init__(self, x_vocab_chars, y_vocab_chars, device, specials_in_x=False):
		specials_y = [_SOS, _EOS, _PAD, _HAL]
		if specials_in_x:
			specials_x = specials_y
		else:
			specials_x = [_PAD]
		self.x_vocab = vocab(
			OrderedDict([(c, 1) for c in x_vocab_chars]),
			specials=specials_x,
			special_first=False)
		self.y_vocab = vocab(
			OrderedDict([(c, 1) for c in y_vocab_chars]),
			specials=specials_y,
			special_first=False)
		self.x_vocab_trans = VocabTransform(self.x_vocab)
		self.y_vocab_trans = VocabTransform(self.y_vocab)
		self.x_to_tensor_trans = ToTensor(padding_value=self.x_vocab[_PAD])
		self.y_to_tensor_trans = ToTensor(padding_value=self.y_vocab[_PAD])
		self.device = device


	def generate_batch(self, bs):
		raise NotImplementedError


	def generate_sample(self):
		raise NotImplementedError


	def get_solution_steps(self, sample: str):
		raise NotImplementedError


	@staticmethod
	def _tokenize_sample(sample: str) -> list:
		return [c for c in sample]
	

	def str_to_batch(self, str_samples, x=True):
		str_samples = [self._tokenize_sample(sample) for sample in str_samples]

		if x:
			tokenized_samples = self.x_vocab_trans(str_samples)
			padded_samples = self.x_to_tensor_trans(tokenized_samples).to(self.device)
			return F.one_hot(padded_samples, num_classes=len(self.x_vocab)).type(torch.float)
		else:
			tokenized_targets = self.y_vocab_trans(str_samples)
			padded_targets = self.y_to_tensor_trans(tokenized_targets).to(self.device)
			return F.one_hot(padded_targets, num_classes=len(self.y_vocab)).type(torch.float)    


	def batch_to_str(self, batch, x=True):
		vocab = self.x_vocab if x else self.y_vocab
		return [''.join(vocab.lookup_tokens(tokens)) for tokens in batch.argmax(-1).tolist()]
