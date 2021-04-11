import torch
from torch import nn
from models.charCNN import ConvolutionalCharEmbedding
from preprocess.dataset import PAD_INDEX

class RNNEncoder(nn.Module):

	def __init__(self, config, pos_vocab_length, word_vocab_length, char_vocab_length):
		super().__init__()
		self.config = config
		self.mode = 'teacher'

		self.word_project = nn.Embedding(word_vocab_length, config.word_emb_dim)

		# POS-tag embeddings will always be trained from scratch.
		if config.use_pos:
			self.pos_embedding = nn.Embedding(pos_vocab_length, config.pos_emb_dim)
		if config.use_charCNN:
			self.charCNN_embedding = ConvolutionalCharEmbedding(char_vocab_length, PAD_INDEX)
			config.charCNN_dim = self.charCNN_embedding.output_size
			print('charCNN output size', config.charCNN_dim)

		# dropout
		# self.input_word_dropout = nn.Dropout(p=config.teacher_dropout)
		self.pos_dropout = nn.Dropout(p=config.teacher_dropout)
		self.char_dropout = nn.Dropout(p=config.teacher_dropout)
		self.word_dropout = nn.Dropout(p=config.teacher_dropout)
		self.rnn1_dropout = nn.Dropout(p=config.teacher_dropout)
		self.rnn2_dropout = nn.Dropout(p=config.teacher_dropout)
		# self.input_word_dropout_student = nn.Dropout(p=config.student_dropout)
		self.pos_dropout_student = nn.Dropout(p=config.student_dropout)
		self.char_dropout_student = nn.Dropout(p=config.student_dropout)
		self.word_dropout_student = nn.Dropout(p=config.student_dropout)
		self.rnn1_dropout_student = nn.Dropout(p=config.student_dropout)
		self.rnn2_dropout_student = nn.Dropout(p=config.student_dropout)

		rnn_input_dim = config.word_emb_dim
		if self.config.use_phobert:
			rnn_input_dim += config.phobert_dim
		if config.use_pos:
			rnn_input_dim += config.pos_emb_dim
		if config.use_charCNN:
			rnn_input_dim += config.charCNN_dim

		self.rnn_size = config.rnn_size
		self.rnn1 = nn.LSTM(input_size=rnn_input_dim, hidden_size=config.rnn_size, batch_first=True, bidirectional=True)
		self.rnn2 = nn.LSTM(input_size=2*config.rnn_size, hidden_size=config.rnn_size, batch_first=True, bidirectional=True,
												dropout=config.teacher_dropout, num_layers=config.rnn_depth-1)

	def forward_student(self, words, phobert_embs, postags, chars):
		word_emb = self.word_project(words)
		if self.config.use_phobert:
			word_emb = torch.cat([word_emb, phobert_embs], dim=2)
		word_emb = self.word_dropout_student(word_emb)

		if self.config.use_pos:
			pos_emb = self.pos_embedding(postags)
			pos_emb = self.pos_dropout_student(pos_emb)
			word_emb = torch.cat([word_emb, pos_emb], dim=2)

		if self.config.use_charCNN:
			char_emb = self.charCNN_embedding(chars)
			char_emb = self.char_dropout_student(char_emb)
			word_emb = torch.cat([word_emb, char_emb], dim=2)

		rnn1_out, _ = self.rnn1(word_emb)
		rnn1_out = self.rnn1_dropout_student(rnn1_out)
		uni_fw = rnn1_out[:, :, :self.rnn_size]
		uni_bw = rnn1_out[:, :, self.rnn_size:]
		rnn2_out, _ = self.rnn2(rnn1_out)
		rnn2_out = self.rnn2_dropout_student(rnn2_out)
		return rnn1_out, rnn2_out, uni_fw, uni_bw

	def forward(self, words, phobert_embs, postags, chars):
		if self.mode == 'student' and self.training:
			return self.forward_student(words, phobert_embs, postags, chars)
		# if self.training and self.config.use_phobert:
		# 	words = self.input_word_dropout(words)

		# Look up
		# word_emb = self.word_embedding(words)
		word_emb = self.word_project(words)
		if self.config.use_phobert:
			word_emb = torch.cat([word_emb, phobert_embs], dim=2)
		if self.training:
			word_emb = self.word_dropout(word_emb)

		if self.config.use_pos:
			pos_emb = self.pos_embedding(postags)
			if self.training:
				pos_emb = self.pos_dropout(pos_emb)
			word_emb = torch.cat([word_emb, pos_emb], dim=2)

		if self.config.use_charCNN:
			char_emb = self.charCNN_embedding(chars)
			if self.training:
				char_emb = self.char_dropout(char_emb)
			word_emb = torch.cat([word_emb, char_emb], dim=2)

		rnn1_out, _ = self.rnn1(word_emb)
		if self.training:
			rnn1_out = self.rnn1_dropout(rnn1_out)
		uni_fw = rnn1_out[:, :, :self.rnn_size]
		uni_bw = rnn1_out[:, :, self.rnn_size:]
		rnn2_out, _ = self.rnn2(rnn1_out)
		if self.training:
			rnn2_out = self.rnn2_dropout(rnn2_out)
		return rnn1_out, rnn2_out, uni_fw, uni_bw
