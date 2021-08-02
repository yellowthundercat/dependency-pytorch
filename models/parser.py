from torch import nn
from models.scorer import BiaffineScorer

from models.utils import get_input_dim, get_dependency_encoder_repr
from utils.chuliu_edmonds import chuliu_edmonds_one_root


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

	def get_scorer_repr(self, words, index_ids, last_index_position, postags, chars, heads, labels, masks, lengths):
		if self.config.encoder == 'biLSTM':
			rnn1_out, rnn2_out, uni_fw, uni_bw, pos_loss = self.encoder(words, index_ids, last_index_position, postags, chars, masks, lengths)
			head_repr, dep_repr = get_dependency_encoder_repr(self.head_repr_type, self.dep_repr_type, rnn1_out, rnn2_out, uni_fw, uni_bw)
		else:
			trans1, trans2, pos_loss = self.encoder(words, index_ids, last_index_position, postags, chars, masks, lengths)
			head_repr = trans1 if self.head_repr_type == 'trans1' else trans2
			dep_repr = trans1 if self.dep_repr_type == 'trans1' else trans2
		loss, preds = self.scorer(head_repr, dep_repr, heads, labels, masks, lengths)
		return loss, preds

	def forward(self, words, index_ids, last_index_position, postags, chars, heads, labels, masks, lengths):
		loss, preds = self.get_scorer_repr(words, index_ids, last_index_position, postags, chars, heads, labels, masks, lengths)
		return loss

	def predict_batch(self, words, index_ids, last_index_position, postags, chars, heads, labels, lengths, masks):
		loss, preds = self.get_scorer_repr(words, index_ids, last_index_position, postags, chars, heads, labels, masks, lengths)
		head_seqs = [chuliu_edmonds_one_root(adj[:l, :l])[1:] for adj, l in zip(preds[0], lengths)]  # remove attachment for the root
		deprel_seqs = [[preds[1][i][j + 1][h] for j, h in enumerate(hs)] for i, hs in enumerate(head_seqs)]
		return head_seqs, deprel_seqs

	def predict_batch_with_loss(self, words, index_ids, last_index_position, postags, chars, heads, labels, pad_masks, lengths):
		loss, preds = self.get_scorer_repr(words, index_ids, last_index_position, postags, chars, heads, labels, pad_masks, lengths)
		head_seqs = [chuliu_edmonds_one_root(adj[:l, :l])[1:] for adj, l in zip(preds[0], lengths)]  # remove attachment for the root
		deprel_seqs = [[preds[1][i][j + 1][h] for j, h in enumerate(hs)] for i, hs in enumerate(head_seqs)]
		return loss, head_seqs, deprel_seqs





