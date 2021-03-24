import torch
from torch import nn

class RNNEncoder(nn.Module):

	def __init__(self, config, pos_vocab_length):
		super().__init__()
		self.config = config

		# self.word_embedding = nn.Embedding(len(word_field.vocab), word_emb_dim)
		# If we're using pre-trained word embeddings, we need to copy them.
		# if word_field.vocab.vectors is not None:
		# 	self.word_embedding.weight = nn.Parameter(word_field.vocab.vectors, requires_grad=update_pretrained)
		self.word_project = nn.Linear(config.phobert_dim + config.attention_emb_dim, config.word_emb_dim)

		# POS-tag embeddings will always be trained from scratch.
		self.pos_embedding = nn.Embedding(pos_vocab_length, config.pos_emb_dim)

		# dropout
		self.input_word_dropout = nn.Dropout(p=config.input_dropout)
		self.pos_dropout = nn.Dropout(p=config.input_dropout)
		self.word_dropout = nn.Dropout(p=config.word_dropout)
		self.rnn1_dropout = nn.Dropout(p=config.rnn_dropout)
		self.rnn2_dropout = nn.Dropout(p=config.rnn_dropout)

		rnn_input_dim = config.word_emb_dim + config.pos_emb_dim
		if not config.use_pos:
			rnn_input_dim = config.word_emb_dim

		self.rnn_size = config.rnn_size
		self.rnn1 = nn.LSTM(input_size=rnn_input_dim, hidden_size=config.rnn_size, batch_first=True, bidirectional=True)
		self.rnn2 = nn.LSTM(input_size=2*config.rnn_size, hidden_size=config.rnn_size, batch_first=True, bidirectional=True,
												dropout=config.rnn_dropout, num_layers=config.rnn_depth-1)

	def forward(self, words, postags):
		if self.training:
			words = self.input_word_dropout(words)
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

		rnn1_out, _ = self.rnn1(word_emb)
		uni_fw = rnn1_out[:, :, :self.rnn_size]
		uni_bw = rnn1_out[:, :, self.rnn_size:]
		if self.training:
			rnn1_out = self.rnn1_dropout(rnn1_out)
		rnn2_out, _ = self.rnn2(rnn1_out)
		if self.training:
			rnn2_out = self.rnn2_dropout(rnn2_out)
		return rnn1_out, rnn2_out, uni_fw, uni_bw
