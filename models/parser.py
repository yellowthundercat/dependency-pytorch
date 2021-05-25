import torch

from torch import nn
from models.scorer import BiaffineScorer

from utils.mst import mst
from models.utils import get_input_dim, get_dependency_encoder_repr

class Parser(nn.Module):

	def __init__(self, encoder, n_label, config, head_repr_type, dep_repr_type, dropout):
		super().__init__()
		self.head_repr_type = head_repr_type
		self.dep_repr_type = dep_repr_type
		self.config = config

		# Sentence encoder module.
		self.encoder = encoder
		scorer_size = get_input_dim(config, head_repr_type)
		self.scorer = BiaffineScorer(config, scorer_size, n_label, dropout)

		# Loss function that we will use during training.
		self.loss = torch.nn.CrossEntropyLoss(reduction='none')

	def get_scorer_repr(self, words, index_ids, last_index_position, postags, chars, masks, lengths):
		if self.config.encoder == 'biLSTM':
			rnn1_out, rnn2_out, uni_fw, uni_bw, pos_loss = self.encoder(words, index_ids, last_index_position, postags, chars, masks, lengths)
			head_repr, dep_repr = get_dependency_encoder_repr(self.head_repr_type, self.dep_repr_type, rnn1_out, rnn2_out, uni_fw, uni_bw)
		else:
			trans1, trans2, pos_loss = self.encoder(words, index_ids, last_index_position, postags, chars, masks, lengths)
			head_repr = trans1 if self.head_repr_type == 'trans1' else trans2
			dep_repr = trans1 if self.dep_repr_type == 'trans1' else trans2
		arc_score, lab_score, lin_scores, dist_kld, head_offset = self.scorer(head_repr, dep_repr)
		return arc_score, lab_score, pos_loss, lin_scores, dist_kld, head_offset

	def forward(self, words, index_ids, last_index_position, postags, chars, heads, labels, masks, lengths):
		arc_score, lab_score, pos_loss, lin_scores, dist_kld, head_offset = self.get_scorer_repr(words, index_ids, last_index_position, postags, chars, masks, lengths)

		# We don't want to evaluate the loss or attachment score for the positions
		# where we have a padding token. So we create a mask that will be zero for those
		# positions and one elsewhere.
		pad_mask = masks
		loss = self.compute_loss(arc_score, lab_score, lin_scores, dist_kld, head_offset, heads, labels, pad_mask)
		if pos_loss is not None:
			loss += pos_loss * self.config.pos_lambda
		return loss

	def compute_loss(self, arc_score, lab_score, lin_scores, dist_kld, head_offset, heads, labels, pad_mask):
		arc_loss = self.arc_loss(arc_score, heads, pad_mask)
		lab_loss = self.lab_loss(lab_score, heads, labels, pad_mask)
		lin_loss = self.lin_loss(lin_scores, heads, pad_mask, head_offset)
		dist_loss = torch.gather(dist_kld[:, :], 2, heads.unsqueeze(2)).view(-1)
		loss = arc_loss + lab_loss + lin_loss - dist_loss
		pad_mask = pad_mask.view(-1)
		avg_loss = loss.dot(pad_mask) / pad_mask.sum()
		return avg_loss

	def lin_loss(self, lin_scores, heads, pad_mask, head_offset):
		lin_scores = torch.gather(lin_scores[:, :], 2, heads.unsqueeze(2)).view(-1)
		lin_scores = torch.cat([-lin_scores.unsqueeze(1) / 2, lin_scores.unsqueeze(1) / 2], 1)
		lin_target = torch.gather((head_offset[:, :] > 0).long(), 2, heads.unsqueeze(2))
		return self.loss(lin_scores.contiguous(), lin_target.view(-1))

	def arc_loss(self, arc_score, heads, pad_mask):
		"""Compute the loss for the arc predictions."""
		arc_score = arc_score.transpose(-1, -2).contiguous()  # [batch, sent_len, sent_len]
		n_sentences, n_words, _ = arc_score.shape
		edge_scores = arc_score.view(n_sentences * n_words, n_words)
		heads = heads.view(n_sentences * n_words)
		loss = self.loss(edge_scores, heads)
		return loss

	def lab_loss(self, lab_score, heads, labels, pad_mask):
		"""Compute the loss for the label predictions on the gold arcs (heads)."""
		heads = heads.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, sent_len]
		heads = heads.expand(-1, lab_score.size(1), -1, -1)  # [batch, n_labels, 1, sent_len]
		lab_score = torch.gather(lab_score, 2, heads).squeeze(2)  # [batch, n_labels, sent_len]
		lab_score = lab_score.transpose(-1, -2)  # [batch, sent_len, n_labels]
		lab_score = lab_score.contiguous().view(-1, lab_score.size(-1))  # [batch*sent_len, n_labels]
		labels = labels.view(-1)  # [batch*sent_len]
		loss = self.loss(lab_score, labels)
		return loss

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

	def predict_batch(self, words, index_ids, last_index_position, postags, chars, lengths, masks):
		arc_score, lab_score, pos_loss, lin_scores, dist_kld, head_offset = self.get_scorer_repr(words, index_ids, last_index_position, postags, chars, masks, lengths)
		return self.parse_from_score(arc_score, lab_score, lengths)

	def predict_batch_with_loss(self, words, index_ids, last_index_position, postags, chars, heads, labels, pad_masks, lengths):
		arc_score, lab_score, pos_loss, lin_scores, dist_kld, head_offset = self.get_scorer_repr(words, index_ids, last_index_position, postags, chars, pad_masks, lengths)
		loss = self.compute_loss(arc_score, lab_score, lin_scores, dist_kld, head_offset, heads, labels, pad_masks)
		head_list, lab_list = self.parse_from_score(arc_score, lab_score, lengths)
		return loss, head_list, lab_list





