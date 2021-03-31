from torch import nn
import numpy as np
import torch
import copy

class HighwayNetwork(nn.Module):
	"""A highway network used in the character convolution word embeddings."""

	def __init__(self, input_size, activation='ReLU'):
		super(HighwayNetwork, self).__init__()
		self.linear = nn.Linear(input_size, input_size)
		self.gate = nn.Linear(input_size, 1)
		self.act_fn = getattr(nn, activation)()
		self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		t = self.sigmoid(self.gate(x))
		out = self.act_fn(self.linear(x))
		return t * out + (1 - t) * x


class ConvolutionalCharEmbedding(nn.Module):
	"""Convolutional character embedding following https://arxiv.org/pdf/1508.06615.pdf."""

	def __init__(self, nchars, padding_idx, emb_dim=15, filter_factor=25, activation='Tanh'):
		super(ConvolutionalCharEmbedding, self).__init__()
		self.padding_idx = padding_idx
		self.embedding = nn.Embedding(nchars, emb_dim, padding_idx=padding_idx)

		filter_size = lambda kernel_size: filter_factor * kernel_size
		self.output_size = sum(map(filter_size, range(1, 4)))  # max kernel size + 1
		self.conv1 = nn.Conv1d(emb_dim, filter_size(1), kernel_size=1)
		self.conv2 = nn.Conv1d(emb_dim, filter_size(2), kernel_size=2)
		self.conv3 = nn.Conv1d(emb_dim, filter_size(3), kernel_size=3)
		# self.conv4 = nn.Conv1d(emb_dim, filter_size(4), kernel_size=4)
		# self.conv5 = nn.Conv1d(emb_dim, filter_size(5), kernel_size=5)
		# self.conv6 = nn.Conv1d(emb_dim, filter_size(6), kernel_size=6)

		self.act_fn = getattr(nn, activation)()

		self.pool = nn.AdaptiveMaxPool1d(1)  # Max pooling over time.

		self.highway = HighwayNetwork(self.output_size)

	def forward(self, x):
		"""Expect input of shape (batch, sent_len, word_len)."""
		# Preprocessing of character batch.
		batch_size, sent_len, word_len = x.shape
		x = x.view(-1, word_len)  # (batch * sent, word)
		mask = (x != self.padding_idx).float()
		x = self.embedding(x)  # (batch * sent, word, emb)
		mask = mask.unsqueeze(-1).repeat(1, 1, x.size(-1))
		x = mask * x
		x = x.transpose(1, 2)  # (batch * sent, emb, word)

		# Ready for input
		f1 = self.pool(self.act_fn(self.conv1(x))).squeeze(-1)
		f2 = self.pool(self.act_fn(self.conv2(x))).squeeze(-1)
		f3 = self.pool(self.act_fn(self.conv3(x))).squeeze(-1)
		# f4 = self.pool(self.act_fn(self.conv4(x))).squeeze(-1)
		# f5 = self.pool(self.act_fn(self.conv5(x))).squeeze(-1)
		# f6 = self.pool(self.act_fn(self.conv6(x))).squeeze(-1)

		# f = torch.cat([f1, f2, f3, f4, f5, f6], dim=-1)
		f = torch.cat([f1, f2, f3], dim=-1)

		f = self.highway(f)

		return f.contiguous().view(batch_size, sent_len, f.size(-1))  # (batch, sent, emb)

	@property
	def num_parameters(self):
		"""Returns the number of trainable parameters of the model."""
		return sum(np.prod(p.shape) for p in self.parameters() if p.requires_grad)
