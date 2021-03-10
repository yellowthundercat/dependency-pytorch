import torch

from torch import nn
from models.scorer import BiaffineScorer
from models.encoder import RNNEncoder

class Parser(nn.Module):

	def __init__(self, pos_vocab_length, n_label, config, word_emb_dim, pos_emb_dim, rnn_size, rnn_depth, mlp_size, update_pretrained=False):
		super().__init__()
		self.config = config

		# Sentence encoder module.
		self.encoder = RNNEncoder(config, word_emb_dim, pos_vocab_length, pos_emb_dim, rnn_size, rnn_depth, update_pretrained)

		# Edge scoring module.
		self.scorer = BiaffineScorer(2 * rnn_size, mlp_size, n_label)

		# Loss function that we will use during training.
		self.loss = torch.nn.CrossEntropyLoss(reduction='none')

	def word_tag_dropout(self, words, postags, p_drop):
		# Randomly replace some of the positions in the word and postag tensors with a zero.
		# This solution is a bit hacky because we assume that zero corresponds to the "unknown" token.
		w_dropout_mask = (torch.rand(size=words.shape, device=words.device) > p_drop).long()
		p_dropout_mask = (torch.rand(size=postags.shape, device=words.device) > p_drop).long()
		return words * w_dropout_mask, postags * p_dropout_mask

	def forward(self, words, postags, heads, labels, masks, evaluate=False):

		if self.training:
			# If we are training, apply the word/tag dropout to the word and tag tensors.
			words, postags = self.word_tag_dropout(words, postags, self.config.drop_out_rate)

		encoded = self.encoder(words, postags)
		arc_score, lab_score = self.scorer(encoded)

		# We don't want to evaluate the loss or attachment score for the positions
		# where we have a padding token. So we create a mask that will be zero for those
		# positions and one elsewhere.
		# pad_mask = (words != self.pad_id).float()
		pad_mask = masks

		arc_loss = self.arc_loss(arc_score, heads, pad_mask)
		lab_loss = self.lab_loss(lab_score, heads, labels, pad_mask)
		loss = arc_loss + lab_loss

		if evaluate:
			n_errors, n_tokens = self.evaluate(arc_score, heads, pad_mask)
			return loss, n_errors, n_tokens
		else:
			return loss

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

	def evaluate(self, arc_score, heads, pad_mask):
		arc_score = arc_score.transpose(-1, -2).contiguous()
		n_sentences, n_words, _ = arc_score.shape
		arc_score = arc_score.view(n_sentences * n_words, n_words)
		heads = heads.view(n_sentences * n_words)
		pad_mask = pad_mask.view(n_sentences * n_words)
		n_tokens = pad_mask.sum()
		predictions = arc_score.argmax(dim=1)
		n_errors = (predictions != heads).float().dot(pad_mask)
		return n_errors.item(), n_tokens.item()

	def predict(self, words, postags):
		# This method is used to parse a sentence when the model has been trained.
		encoded = self.encoder(words, postags)
		arc_score = self.scorer(encoded)
		arc_score = arc_score.transpose(-1, -2).contiguous()
		return arc_score.argmax(dim=2)
