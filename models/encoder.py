import torch
from torch import nn
from models.charCNN import ConvolutionalCharEmbedding
from models.other_transformer import TransformerEncoder
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
		self.word_dropout = nn.Dropout(p=config.teacher_dropout)
		self.rnn1_dropout = nn.Dropout(p=config.teacher_dropout)
		self.rnn2_dropout = nn.Dropout(p=config.teacher_dropout)
		self.word_dropout_student = nn.Dropout(p=config.student_dropout)
		self.rnn1_dropout_student = nn.Dropout(p=config.student_dropout)
		self.rnn2_dropout_student = nn.Dropout(p=config.student_dropout)

		input_dim = config.word_emb_dim
		if self.config.use_phobert:
			input_dim += config.phobert_dim
		if config.use_pos:
			input_dim += config.pos_emb_dim
		if config.use_charCNN:
			input_dim += config.charCNN_dim

		if config.encoder == 'biLSTM':
			self.rnn_size = config.rnn_size
			self.rnn1 = nn.LSTM(input_size=input_dim, hidden_size=config.rnn_size, batch_first=True, bidirectional=True)
			self.rnn2 = nn.LSTM(input_size=2*config.rnn_size, hidden_size=config.rnn_size, batch_first=True, bidirectional=True,
													dropout=config.teacher_dropout, num_layers=config.rnn_depth-1)
		else:
			self.transformer = TransformerEncoder(input_dim, config.transformer_layer, config.transformer_dim,
																						config.transformer_ff_dim, config.transformer_head, config.transformer_dropout)

	def forward_student_training(self, words, phobert_embs, postags, chars):
		if self.config.encoder == 'transformer':
			aux = (words != PAD_INDEX).unsqueeze(-2)  # mask
		word_emb = self.word_project(words)
		if self.config.use_phobert:
			word_emb = torch.cat([word_emb, phobert_embs], dim=2)

		if self.config.use_pos:
			pos_emb = self.pos_embedding(postags)
			word_emb = torch.cat([word_emb, pos_emb], dim=2)

		if self.config.use_charCNN:
			char_emb = self.charCNN_embedding(chars)
			word_emb = torch.cat([word_emb, char_emb], dim=2)

		word_emb = self.word_dropout_student(word_emb)

		if self.config.encoder == 'biLSTM':
			rnn1_out, _ = self.rnn1(word_emb)
			rnn1_out = self.rnn1_dropout_student(rnn1_out)
			uni_fw = rnn1_out[:, :, :self.rnn_size]
			uni_bw = rnn1_out[:, :, self.rnn_size:]
			rnn2_out, _ = self.rnn2(rnn1_out)
			rnn2_out = self.rnn2_dropout_student(rnn2_out)
			return rnn1_out, rnn2_out, uni_fw, uni_bw
		else:
			return self.transformer(word_emb, aux)

	def forward(self, words, phobert_embs, postags, chars):
		if self.mode == 'student' and self.training:
			return self.forward_student_training(words, phobert_embs, postags, chars)

		# Look up
		if self.config.encoder == 'transformer':
			aux = (words != PAD_INDEX).unsqueeze(-2)  # mask
		word_emb = self.word_project(words)
		if self.config.use_phobert:
			word_emb = torch.cat([word_emb, phobert_embs], dim=2)

		if self.config.use_pos:
			pos_emb = self.pos_embedding(postags)
			word_emb = torch.cat([word_emb, pos_emb], dim=2)

		if self.config.use_charCNN:
			char_emb = self.charCNN_embedding(chars)
			word_emb = torch.cat([word_emb, char_emb], dim=2)
		word_emb = self.word_dropout(word_emb)

		if self.config.encoder == 'biLSTM':
			rnn1_out, _ = self.rnn1(word_emb)
			rnn1_out = self.rnn1_dropout(rnn1_out)
			uni_fw = rnn1_out[:, :, :self.rnn_size]
			uni_bw = rnn1_out[:, :, self.rnn_size:]
			rnn2_out, _ = self.rnn2(rnn1_out)
			rnn2_out = self.rnn2_dropout(rnn2_out)
			return rnn1_out, rnn2_out, uni_fw, uni_bw
		else:
			return self.transformer(word_emb, aux)
