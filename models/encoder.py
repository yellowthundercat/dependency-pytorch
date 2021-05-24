import torch
from torch import nn
from models.charCNN import ConvolutionalCharEmbedding
from models.pos_scorer import Pos_scorer
from models.other_transformer import TransformerEncoder
from preprocess.dataset import PAD_INDEX

class Encoder(nn.Module):

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
		self.word_dropout = nn.Dropout(p=config.teacher_dropout)
		self.pos_dropout = nn.Dropout(p=config.teacher_dropout)
		self.rnn1_dropout = nn.Dropout(p=config.teacher_dropout)
		self.rnn2_dropout = nn.Dropout(p=config.teacher_dropout)
		self.word_dropout_student = nn.Dropout(p=config.student_dropout)
		self.pos_dropout_student = nn.Dropout(p=config.student_dropout)
		self.rnn1_dropout_student = nn.Dropout(p=config.student_dropout)
		self.rnn2_dropout_student = nn.Dropout(p=config.student_dropout)

		input_dim = 0
		if self.config.use_word_emb_scratch:
			input_dim += config.word_emb_dim
		if self.config.use_phobert:
			input_dim += config.phobert_dim
		if config.use_pos:
			input_dim += config.pos_emb_dim
		if config.use_charCNN:
			input_dim += config.charCNN_dim

		if config.encoder == 'biLSTM':
			self.rnn_size = config.rnn_size
			self.rnn1 = nn.LSTM(input_size=input_dim, hidden_size=config.rnn_size, batch_first=True, bidirectional=True,
													dropout=config.teacher_dropout, num_layers=config.rnn_1_depth)
			if config.rnn_2_depth > 0:
				rnn2_input_dim = 2*config.rnn_size
				self.rnn2 = nn.LSTM(input_size=rnn2_input_dim, hidden_size=config.rnn_size, batch_first=True, bidirectional=True,
														dropout=config.teacher_dropout, num_layers=config.rnn_2_depth)
		else:
			self.transformer1 = TransformerEncoder(input_dim, config.transformer_1_depth, config.transformer_dim,
												   config.transformer_ff_dim, config.transformer_head, config.transformer_dropout)
			if config.transformer_2_depth > 0:
				trans2_input_dim = config.transformer_dim
				self.transformer2 = TransformerEncoder(trans2_input_dim, config.transformer_2_depth, config.transformer_dim,
																						 config.transformer_ff_dim, config.transformer_head, config.transformer_dropout)

	def forward_emb(self, words, phobert_embs, postags, chars):
		# Look up
		word_emb = None
		if self.config.use_word_emb_scratch:
			word_emb = self.word_project(words)
		if self.config.use_phobert:
			if word_emb is not None:
				word_emb = torch.cat([word_emb, phobert_embs], dim=2)
			else:
				word_emb = phobert_embs

		if self.config.use_charCNN:
			char_emb = self.charCNN_embedding(chars)
			word_emb = torch.cat([word_emb, char_emb], dim=2)
		if self.mode == 'teacher':
			word_emb = self.word_dropout(word_emb)
		else:
			word_emb = self.word_dropout_student(word_emb)

		if self.config.use_pos:
			pos_emb = self.pos_embedding(postags)
			if self.mode == 'teacher':
				pos_emb = self.pos_dropout(pos_emb)
			else:
				pos_emb = self.pos_dropout_student(pos_emb)
			word_emb = torch.cat([word_emb, pos_emb], dim=2)
		return word_emb

	def forward(self, words, phobert_embs, postags, chars, masks):
		word_emb = self.forward_emb(words, phobert_embs, postags, chars)

		pos_loss = None
		if self.config.encoder == 'biLSTM':
			rnn1_out, _ = self.rnn1(word_emb)
			if self.mode == 'teacher':
				rnn1_out = self.rnn1_dropout(rnn1_out)
			else:
				rnn1_out = self.rnn1_dropout_student(rnn1_out)
			uni_fw = rnn1_out[:, :, :self.rnn_size]
			uni_bw = rnn1_out[:, :, self.rnn_size:]

			if self.config.rnn_2_depth > 0:
				rnn2_in = rnn1_out
				rnn2_out, _ = self.rnn2(rnn2_in)
				if self.mode == 'teacher':
					rnn2_out = self.rnn2_dropout(rnn2_out)
				else:
					rnn2_out = self.rnn2_dropout_student(rnn2_out)
				return rnn1_out, rnn2_out, uni_fw, uni_bw, pos_loss
			return rnn1_out, rnn1_out, uni_fw, uni_bw, pos_loss
		else:
			aux = (words != PAD_INDEX).unsqueeze(-2)  # mask
			trans1_out = self.transformer1(word_emb, aux)

			if self.config.transformer_2_depth > 0:
				trans2_in = trans1_out
				trans2_out = self.transformer2(trans2_in, aux)
				return trans1_out, trans2_out, pos_loss
			return trans1_out, trans1_out, pos_loss
