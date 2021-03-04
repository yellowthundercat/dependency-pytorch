import torch

from torch import nn
from models.edge_scorer import BiaffineEdgeScorer
from models.encoder import RNNEncoder

class EdgeFactoredParser(nn.Module):

	def __init__(self, pos_vocab_length, config, word_emb_dim, pos_emb_dim, rnn_size, rnn_depth, mlp_size, update_pretrained=False):
		super().__init__()
		self.config = config

		# Sentence encoder module.
		self.encoder = RNNEncoder(word_emb_dim, pos_vocab_length, pos_emb_dim, rnn_size, rnn_depth, update_pretrained)

		# Edge scoring module.
		self.edge_scorer = BiaffineEdgeScorer(2 * rnn_size, mlp_size)

		# To deal with the padding positions later, we need to know the
		# encoding of the padding dummy word.
		# self.pad_id = word_field.vocab.stoi[word_field.pad_token]

		# Loss function that we will use during training.
		self.loss = torch.nn.CrossEntropyLoss(reduction='none')

	def word_tag_dropout(self, words, postags, p_drop):
		# Randomly replace some of the positions in the word and postag tensors with a zero.
		# This solution is a bit hacky because we assume that zero corresponds to the "unknown" token.
		w_dropout_mask = (torch.rand(size=words.shape, device=words.device) > p_drop).long()
		p_dropout_mask = (torch.rand(size=words.shape, device=words.device) > p_drop).long()
		return words * w_dropout_mask, postags * p_dropout_mask

	def forward(self, words, postags, heads, labels, masks, evaluate=False):

		if self.training:
			# If we are training, apply the word/tag dropout to the word and tag tensors.
			words, postags = self.word_tag_dropout(words, postags, self.config.drop_out_rate)

		encoded = self.encoder(words, postags)
		edge_scores = self.edge_scorer(encoded)

		# We don't want to evaluate the loss or attachment score for the positions
		# where we have a padding token. So we create a mask that will be zero for those
		# positions and one elsewhere.
		# pad_mask = (words != self.pad_id).float()
		pad_mask = masks

		loss = self.compute_loss(edge_scores, heads, pad_mask)

		if evaluate:
			n_errors, n_tokens = self.evaluate(edge_scores, heads, pad_mask)
			return loss, n_errors, n_tokens
		else:
			return loss

	def compute_loss(self, edge_scores, heads, pad_mask):
		n_sentences, n_words, _ = edge_scores.shape
		edge_scores = edge_scores.view(n_sentences * n_words, n_words)
		heads = heads.view(n_sentences * n_words)
		pad_mask = pad_mask.view(n_sentences * n_words)
		loss = self.loss(edge_scores, heads)
		avg_loss = loss.dot(pad_mask) / pad_mask.sum()
		return avg_loss

	def evaluate(self, edge_scores, heads, pad_mask):
		n_sentences, n_words, _ = edge_scores.shape
		edge_scores = edge_scores.view(n_sentences * n_words, n_words)
		heads = heads.view(n_sentences * n_words)
		pad_mask = pad_mask.view(n_sentences * n_words)
		n_tokens = pad_mask.sum()
		predictions = edge_scores.argmax(dim=1)
		n_errors = (predictions != heads).float().dot(pad_mask)
		return n_errors.item(), n_tokens.item()

	def predict(self, words, postags):
		# This method is used to parse a sentence when the model has been trained.
		encoded = self.encoder(words, postags)
		edge_scores = self.edge_scorer(encoded)
		return edge_scores.argmax(dim=2)
