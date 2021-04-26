from torch import nn
import torch
from models.utils import get_input_dim, get_repr_from_mode

class Pos_paser(nn.Module):
	def __init__(self, config, repr_type, encoder, is_activate, vocab_pos_length, dropout):
		super().__init__()
		self.config = config
		self.repr_type = repr_type
		self.encoder = encoder
		input_size = get_input_dim(config, repr_type)
		if is_activate:
			self.project = nn.Sequential(nn.Linear(input_size, config.pos_hidden_dim), nn.ReLU())
		else:
			self.project = nn.Linear(input_size, config.pos_hidden_dim)
		self.scorer = nn.Linear(config.pos_hidden_dim, vocab_pos_length)

		self.dropout = nn.Dropout(p=dropout)
		self.loss = torch.nn.CrossEntropyLoss(reduction='none')

	def forward(self, words, phobert_embs, postags, chars, masks):
		if self.config.encoder == 'biLSTM':
			rnn1_out, rnn2_out, uni_fw, uni_bw = self.encoder(words, phobert_embs, postags, chars)
			sen_repr = get_repr_from_mode(rnn1_out, rnn2_out, uni_fw, uni_bw, self.repr_type)
		else:
			sen_repr = self.encoder(words, phobert_embs, postags, chars)
		hid_emb = self.project(sen_repr)
		hid_emb = self.dropout(hid_emb)
		predict = self.scorer(hid_emb)
		return self.compute_loss(predict, postags, masks)

	def compute_loss(self, predict, postags, masks):
		n_sentences, n_words, n_tags = predict.shape
		predict = predict.view(n_sentences * n_words, n_tags)
		postags = postags.view(-1)
		masks = masks.view(-1)
		loss = self.loss(predict, postags)
		avg_loss = loss.dot(masks) / masks.sum()
		return avg_loss
