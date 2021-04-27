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
		self.softmax = nn.LogSoftmax(dim=2)
		self.loss = nn.NLLLoss(reduction='none')

	def get_predict(self, words, phobert_embs, chars):
		if self.config.encoder == 'biLSTM':
			rnn1_out, rnn2_out, uni_fw, uni_bw = self.encoder(words, phobert_embs, [], chars)
			sen_repr = get_repr_from_mode(rnn1_out, rnn2_out, uni_fw, uni_bw, self.repr_type)
		else:
			sen_repr = self.encoder(words, phobert_embs, [], chars)
		hid_emb = self.project(sen_repr)
		hid_emb = self.dropout(hid_emb)
		predict = self.softmax(self.scorer(hid_emb))
		return predict

	def forward(self, words, phobert_embs, postags, chars, masks):
		predict = self.get_predict(words, phobert_embs, chars)
		return self.compute_loss(predict, postags, masks), predict

	def predict(self, words, phobert_embs, chars):
		return self.get_predict(words, phobert_embs, chars)

	def compute_loss(self, predict, postags, masks):
		n_sentences, n_words, n_tags = predict.shape
		predict = predict.view(n_sentences * n_words, n_tags)
		postags = postags.view(-1)
		masks = masks.view(-1)
		loss = self.loss(predict, postags)
		avg_loss = loss.dot(masks) / masks.sum()
		return avg_loss

	def evaluate(self, words, phobert_embs, postags, chars, masks):
		pos_predict = self.get_predict(words, phobert_embs, chars)
		n_sentences, n_words, n_label = pos_predict.shape
		predict = pos_predict.view(n_sentences * n_words, n_label)
		postags = postags.view(n_sentences * n_words)
		pad_mask = masks.view(n_sentences * n_words)
		n_tokens = pad_mask.sum()
		predictions = predict.argmax(dim=1)
		n_errors = (predictions != postags).float().dot(pad_mask)
		return n_tokens.item(), n_errors.item(), pos_predict


