import torch

from torch import nn
from models.scorer import BiaffineScorer

from utils.mst import mst

class Parser(nn.Module):

	def __init__(self, encoder, n_label, config, head_repr_type, dep_repr_type):
		super().__init__()
		self.head_repr_type = head_repr_type
		self.dep_repr_type = dep_repr_type
		self.config = config

		# Sentence encoder module.
		self.encoder = encoder
		# Edge scoring module.
		rnn_size = 4*config.rnn_size
		if head_repr_type != 'uni_bi':
			rnn_size = config.rnn_size
		self.scorer = BiaffineScorer(config, rnn_size, n_label)

		# Loss function that we will use during training.
		self.loss = torch.nn.CrossEntropyLoss(reduction='none')

	# def word_tag_dropout(self, words, postags, p_drop):
	# 	# Randomly replace some of the positions in the word and postag tensors with a zero.
	# 	# This solution is a bit hacky because we assume that zero corresponds to the "unknown" token.
	# 	w_dropout_mask = (torch.rand(size=words.shape, device=words.device) > p_drop).long()
	# 	p_dropout_mask = (torch.rand(size=postags.shape, device=words.device) > p_drop).long()
	# 	return words * w_dropout_mask, postags * p_dropout_mask

	def get_repr_from_mode(self, rnn1_out, rnn2_out, uni_fw, uni_bw, mode):
		if mode == 'uni_bi':
			return torch.cat([rnn1_out, rnn2_out], dim=2)
		if mode == 'uni_fw':
			return uni_fw
		return uni_bw

	def get_encoder_repr(self, rnn1_out, rnn2_out, uni_fw, uni_bw):
		head_repr = self.get_repr_from_mode(rnn1_out, rnn2_out, uni_fw, uni_bw, self.head_repr_type)
		dep_repr = self.get_repr_from_mode(rnn1_out, rnn2_out, uni_fw, uni_bw, self.dep_repr_type)
		return head_repr, dep_repr

	def forward(self, words, postags, heads, labels, masks):

		# if self.training:
		# 	# If we are training, apply the word/tag dropout to the word and tag tensors.
		# 	words, postags = self.word_tag_dropout(words, postags, self.config.drop_out_rate)

		rnn1_out, rnn2_out, uni_fw, uni_bw = self.encoder(words, postags)
		head_repr, dep_repr = self.get_encoder_repr(rnn1_out, rnn2_out, uni_fw, uni_bw)
		arc_score, lab_score = self.scorer(head_repr, dep_repr)

		# We don't want to evaluate the loss or attachment score for the positions
		# where we have a padding token. So we create a mask that will be zero for those
		# positions and one elsewhere.
		pad_mask = masks

		return self.compute_loss(arc_score, lab_score, heads, labels, pad_mask)

	def compute_loss(self, arc_score, lab_score, heads, labels, pad_mask):
		arc_loss = self.arc_loss(arc_score, heads, pad_mask)
		lab_loss = self.lab_loss(lab_score, heads, labels, pad_mask)
		return arc_loss + lab_loss

	def arc_loss(self, arc_score, heads, pad_mask):
		"""Compute the loss for the arc predictions."""
		arc_score = arc_score.transpose(-1, -2).contiguous()  # [batch, sent_len, sent_len]
		n_sentences, n_words, _ = arc_score.shape
		edge_scores = arc_score.view(n_sentences * n_words, n_words)
		heads = heads.view(n_sentences * n_words)
		pad_mask = pad_mask.view(n_sentences * n_words)
		loss = self.loss(edge_scores, heads)
		avg_loss = loss.dot(pad_mask) / pad_mask.sum()
		return avg_loss

	def lab_loss(self, lab_score, heads, labels, pad_mask):
		"""Compute the loss for the label predictions on the gold arcs (heads)."""
		heads = heads.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, sent_len]
		heads = heads.expand(-1, lab_score.size(1), -1, -1)  # [batch, n_labels, 1, sent_len]
		lab_score = torch.gather(lab_score, 2, heads).squeeze(2)  # [batch, n_labels, sent_len]
		lab_score = lab_score.transpose(-1, -2)  # [batch, sent_len, n_labels]
		lab_score = lab_score.contiguous().view(-1, lab_score.size(-1))  # [batch*sent_len, n_labels]
		labels = labels.view(-1)  # [batch*sent_len]
		pad_mask = pad_mask.view(-1)
		loss = self.loss(lab_score, labels)
		avg_loss = loss.dot(pad_mask) / pad_mask.sum()
		return avg_loss

	def parse_from_score(self, arc_score, lab_score, lengths):
		arc_score = arc_score.cpu()
		lab_score = lab_score.cpu()
		head_list = []
		lab_list = []
		for index, sentence_length in enumerate(lengths):
			sent_arc_score = arc_score[index].data.numpy()[:sentence_length, :sentence_length]
			heads = mst(sent_arc_score)
			sent_lab_score = lab_score[index]
			select = torch.LongTensor(heads).unsqueeze(0).expand(sent_lab_score.size(0), -1)
			select = torch.autograd.Variable(select)
			selected = torch.gather(sent_lab_score, 1, select.unsqueeze(1)).squeeze(1)
			_, labels = selected.max(dim=0)
			labels = labels.data.numpy()
			head_list.append(heads)
			lab_list.append(labels)
		return head_list, lab_list

	def predict_batch(self, words, postags, lengths):
		rnn1_out, rnn2_out, uni_fw, uni_bw = self.encoder(words, postags)
		head_repr, dep_repr = self.get_encoder_repr(rnn1_out, rnn2_out, uni_fw, uni_bw)
		arc_score, lab_score = self.scorer(head_repr, dep_repr)
		return self.parse_from_score(arc_score, lab_score, lengths)

	def predict_batch_with_loss(self, words, postags, heads, labels, pad_masks, lengths):
		rnn1_out, rnn2_out, uni_fw, uni_bw = self.encoder(words, postags)
		head_repr, dep_repr = self.get_encoder_repr(rnn1_out, rnn2_out, uni_fw, uni_bw)
		arc_score, lab_score = self.scorer(head_repr, dep_repr)
		loss = self.compute_loss(arc_score, lab_score, heads, labels, pad_masks)
		head_list, lab_list = self.parse_from_score(arc_score, lab_score, lengths)
		return loss, head_list, lab_list





