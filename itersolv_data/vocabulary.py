import torch
from collections import OrderedDict
import torch.nn.functional as F
from torchtext.vocab import vocab
from torchtext.transforms import ToTensor, VocabTransform


EOS = '.'
SOS = '?'
PAD = '#'
SEP = '/'
HAL = '$'


class Vocabulary:

	def __init__(self, x_vocab_chars, y_vocab_chars, device, sos, eos, specials_in_x=False):
		specials_y = [SOS, EOS, PAD, HAL]
		if specials_in_x:
			specials_x = specials_y
		else:
			specials_x = [PAD]
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
		self.x_to_tensor_trans = ToTensor(padding_value=self.x_vocab[PAD])
		self.y_to_tensor_trans = ToTensor(padding_value=self.y_vocab[PAD])
		self.device = device
		self.sos = sos
		self.eos = eos

	@staticmethod
	def _tokenize_sample(sample: str) -> list:
		return [c for c in sample]
	
	def str_to_batch(self, str_samples, x=True):
		if not x:
			if self.sos:
				str_samples = [f"{SOS}{sample}" for sample in str_samples]
			if self.eos:
				str_samples = [f"{sample}{EOS}" for sample in str_samples]
			
		string_tokenized_samples = [self._tokenize_sample(sample) for sample in str_samples]

		if x:
			idx_tokenized_samples = self.x_vocab_trans(string_tokenized_samples)
			idx_padded_samples = self.x_to_tensor_trans(idx_tokenized_samples).to(self.device)
			return idx_padded_samples
		else:
			idx_tokenized_targets = self.y_vocab_trans(string_tokenized_samples)
			idx_padded_targets = self.y_to_tensor_trans(idx_tokenized_targets).to(self.device)
			return idx_padded_targets

	def batch_to_str(self, batch, x=True):
		vocab = self.x_vocab if x else self.y_vocab
		return [''.join(vocab.lookup_tokens(tokens)).replace(PAD, '') for tokens in batch.tolist()]
