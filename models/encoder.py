import torch
from torch import nn

class RNNEncoder(nn.Module):

	def __init__(self, config, word_emb_dim, pos_vocab_length, pos_emb_dim, rnn_size, rnn_depth, update_pretrained):
		super().__init__()
		self.config = config

		# self.word_embedding = nn.Embedding(len(word_field.vocab), word_emb_dim)
		# If we're using pre-trained word embeddings, we need to copy them.
		# if word_field.vocab.vectors is not None:
		# 	self.word_embedding.weight = nn.Parameter(word_field.vocab.vectors, requires_grad=update_pretrained)
		self.word_project = nn.Linear(config.phobert_dim, word_emb_dim)

		# POS-tag embeddings will always be trained from scratch.
		self.pos_embedding = nn.Embedding(pos_vocab_length, pos_emb_dim)

		# dropout
		self.word_dropout = nn.Dropout(p=config.word_dropout)
		self.pos_dropout = nn.Dropout(p=config.pos_dropout)
		self.rnn_dropout = nn.Dropout(p=config.rnn_dropout)

		rnn_input_dim = word_emb_dim + pos_emb_dim
		if not config.use_pos:
			rnn_input_dim = word_emb_dim
		self.rnn = nn.LSTM(input_size=rnn_input_dim, hidden_size=rnn_size, batch_first=True, bidirectional=True, num_layers=rnn_depth)

	def forward(self, words, postags):
		# Look up
		# word_emb = self.word_embedding(words)
		word_emb = self.word_project(words)
		if self.training:
			word_emb = self.word_dropout(word_emb)

		if self.config.use_pos:
			pos_emb = self.pos_embedding(postags)
			if self.training:
				pos_emb = self.pos_dropout(pos_emb)
			word_emb = torch.cat([word_emb, pos_emb], dim=2)

		rnn_out, _ = self.rnn(word_emb)
		if self.training:
			rnn_out = self.rnn_dropout(rnn_out)
		return rnn_out
