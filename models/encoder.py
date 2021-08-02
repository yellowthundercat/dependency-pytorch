import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pack_sequence, PackedSequence
import numpy as np
from models.charCNN import ConvolutionalCharEmbedding
from models.other_transformer import TransformerEncoder
from preprocess.dataset import PAD_INDEX
from models.dropout import WordDropout
from models.hlstm import HighwayLSTM

class Encoder(nn.Module):

	def __init__(self, config, pos_vocab_length, word_vocab_length, char_vocab_length):
		super().__init__()
		self.config = config
		self.mode = 'teacher'

		self.word_project = nn.Embedding(word_vocab_length, config.word_emb_dim)

		# if config.use_phobert:
		# 	self.phobert = AutoModel.from_pretrained("vinai/phobert-base", output_hidden_states=True)
		if config.use_pos:
			self.pos_embedding = nn.Embedding(pos_vocab_length, config.pos_emb_dim)
		if config.use_charCNN:
			self.charCNN_embedding = ConvolutionalCharEmbedding(char_vocab_length, PAD_INDEX)
			config.charCNN_dim = self.charCNN_embedding.output_size
			print('charCNN output size', config.charCNN_dim)

		# dropout
		self.dropout = nn.Dropout(p=config.teacher_dropout)
		self.dropout_student = nn.Dropout(p=config.student_dropout)
		self.worddrop = WordDropout(config.word_dropout)
		self.worddrop_student = WordDropout(config.word_dropout_student)

		input_dim = 0
		if self.config.use_word_emb_scratch:
			input_dim += config.word_emb_dim
		if self.config.use_phobert:
			input_dim += config.phobert_dim
		if config.use_pos:
			input_dim += config.pos_emb_dim
		if config.use_charCNN:
			input_dim += config.charCNN_dim

		self.drop_replacement = nn.Parameter(torch.randn(input_dim) / np.sqrt(input_dim))
		if config.encoder == 'biLSTM':
			self.input_rnn1_size = input_dim
			self.rnn_size = config.rnn_size
			self.parserlstm = HighwayLSTM(input_dim, self.rnn_size, config.rnn_1_depth, batch_first=True, bidirectional=True,
										  dropout=config.teacher_dropout, rec_dropout=config.rec_dropout, highway_func=torch.tanh)
			self.parserlstm_h_init = nn.Parameter(torch.zeros(2 * config.rnn_1_depth, 1, config.rnn_size))
			self.parserlstm_c_init = nn.Parameter(torch.zeros(2 * config.rnn_1_depth, 1, config.rnn_size))
			self.rnn1 = nn.LSTM(input_size=input_dim, hidden_size=config.rnn_size, batch_first=True, bidirectional=True, dropout=config.teacher_dropout, num_layers=config.rnn_1_depth)
			if config.rnn_2_depth > 0:
				rnn2_input_dim = 2*config.rnn_size
				self.parserlstm2 = HighwayLSTM(rnn2_input_dim, self.rnn_size, config.rnn_2_depth, batch_first=True,
											  bidirectional=True, dropout=config.teacher_dropout, rec_dropout=config.rec_dropout, highway_func=torch.tanh)
				self.drop_replacement2 = nn.Parameter(torch.randn(rnn2_input_dim) / np.sqrt(rnn2_input_dim))
				self.parserlstm_h_init2 = nn.Parameter(torch.zeros(2 * config.rnn_2_depth, 1, config.rnn_size))
				self.parserlstm_c_init2 = nn.Parameter(torch.zeros(2 * config.rnn_2_depth, 1, config.rnn_size))
		else:
			self.transformer1 = TransformerEncoder(input_dim, config.transformer_1_depth, config.transformer_dim,
												   config.transformer_ff_dim, config.transformer_head, config.transformer_dropout)
			if config.transformer_2_depth > 0:
				trans2_input_dim = config.transformer_dim
				self.transformer2 = TransformerEncoder(trans2_input_dim, config.transformer_2_depth, config.transformer_dim,
																						 config.transformer_ff_dim, config.transformer_head, config.transformer_dropout)

	def forward_emb(self, words, index_ids, last_index_position, postags, chars, lengths):
		def pack(x):
			return pack_padded_sequence(x, lengths, batch_first=True)
		# Look up
		inputs = []
		if self.config.use_word_emb_scratch:
			word_emb = self.word_project(words)
			inputs += [pack(word_emb)]

		if self.config.use_phobert:
			# when not finetune index_ids is phobert embedding

			# origin_features = self.phobert(index_ids)
			# features = origin_features[2][self.config.phobert_layer]
			# phobert_embs = []
			# for sentence_index in range(last_index_position.size(0)):
			# 	phobert_embedding = []
			# 	last_index_position_list = last_index_position[sentence_index]
			# 	for word_index in range(last_index_position.size(1) - 2):
			# 		start_index = last_index_position_list[word_index]
			# 		end_index = last_index_position_list[word_index + 1]
			# 		if end_index > start_index:
			# 			if self.config.phobert_subword == 'first':
			# 				end_index = start_index + 1
			# 			one_emb = features[sentence_index][start_index:end_index]
			# 			# one_emb = torch.sum(one_emb, 0).cpu().data.numpy().tolist()
			# 			one_emb = torch.sum(one_emb, 0)
			# 		else:
			# 			one_emb = torch.zeros(self.config.phobert_dim, device=self.device)
			# 		phobert_embedding.append(one_emb)
			# 	phobert_embs.append(torch.stack(phobert_embedding, dim=0))
			phobert_embs = index_ids
			inputs += [pack(phobert_embs)]

		if self.config.use_charCNN:
			char_emb = self.charCNN_embedding(chars)
			inputs += [pack(char_emb)]

		if self.config.use_pos:
			pos_emb = self.pos_embedding(postags)
			inputs += [pack(pos_emb)]

		input_batch_size = inputs[0].batch_sizes
		inputs = torch.cat([x.data for x in inputs], 1)
		if self.mode == 'teacher':
			inputs = self.worddrop(inputs, self.drop_replacement)
			inputs = self.dropout(inputs)
		else:
			inputs = self.worddrop_student(inputs, self.drop_replacement)
			inputs = self.dropout_student(inputs)
		return inputs, input_batch_size

	def forward(self, words, index_ids, last_index_position, postags, chars, masks, lengths):

		inputs, input_batch_size = self.forward_emb(words, index_ids, last_index_position, postags, chars, lengths)

		pos_loss = None
		if self.config.encoder == 'biLSTM':
			lstm_inputs = PackedSequence(inputs, input_batch_size)

			rnn1_out, _ = self.parserlstm(lstm_inputs, lengths, hx=(
			self.parserlstm_h_init.expand(2 * self.config.rnn_1_depth, words.size(0), self.rnn_size).contiguous(),
			self.parserlstm_c_init.expand(2 * self.config.rnn_1_depth, words.size(0), self.rnn_size).contiguous()))
			#rnn1_out, _ = self.rnn1(lstm_inputs)

			rnn1_out_padded, _ = pad_packed_sequence(rnn1_out, batch_first=True)
			uni_fw = rnn1_out_padded[:, :, :self.rnn_size]
			uni_bw = rnn1_out_padded[:, :, self.rnn_size:]

			if self.config.rnn_2_depth > 0:
				rnn2_in = rnn1_out_padded
				if self.mode == 'teacher':
					rnn2_in = self.dropout(rnn2_in)
				else:
					rnn2_in = self.dropout_student(rnn2_in)
				rnn2_in = pack_padded_sequence(rnn2_in, lengths, batch_first=True)
				rnn2_out, _ = self.parserlstm2(rnn2_in, lengths, hx=(
				self.parserlstm_h_init2.expand(2 * self.config.rnn_2_depth, words.size(0), self.rnn_size).contiguous(),
				self.parserlstm_c_init2.expand(2 * self.config.rnn_2_depth, words.size(0), self.rnn_size).contiguous()))

				rnn2_out_padded, _ = pad_packed_sequence(rnn2_out, batch_first=True)
				return rnn1_out_padded, rnn2_out_padded, uni_fw, uni_bw, pos_loss

			return rnn1_out_padded, rnn1_out_padded, uni_fw, uni_bw, pos_loss
		else:
			word_emb = pad_packed_sequence(inputs, batch_first=True)
			aux = (words != PAD_INDEX).unsqueeze(-2)  # mask
			trans1_out = self.transformer1(word_emb, aux)
			if self.config.transformer_2_depth > 0:
				trans2_in = trans1_out
				if self.mode == 'teacher':
					trans2_in = self.dropout(trans2_in)
				else:
					trans2_in = self.dropout_student(trans2_in)
				trans2_out = self.transformer2(trans2_in, aux)
				return trans1_out, trans2_out, pos_loss
			return trans1_out, trans1_out, pos_loss
