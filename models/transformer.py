import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math

class PositionalEncoding(nn.Module):
	r"""Inject some information about the relative or absolute position of the tokens
			in the sequence. The positional encodings have the same dimension as
			the embeddings, so that the two can be summed. Here, we use sine and cosine
			functions of different frequencies.
	.. math::
			\text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
			\text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
			\text{where pos is the word position and i is the embed idx)
	Args:
			d_model: the embed dim (required).
			dropout: the dropout value (default=0.1).
			max_len: the max. length of the incoming sequence (default=5000).
	Examples:
			pos_encoder = PositionalEncoding(d_model)
	"""

	def __init__(self, d_model, dropout=0.1, max_len=500):
		super(PositionalEncoding, self).__init__()
		pe = torch.zeros(max_len, d_model)
		position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		pe = pe.unsqueeze(0).transpose(0, 1)
		self.register_buffer('pe', pe)

	def forward(self, x):
		r"""Inputs of forward function
		Args:
				x: the sequence fed to the positional encoder model (required).
		Shape:
				x: [sequence length, batch size, embed dim]
				output: [sequence length, batch size, embed dim]
		Examples:
				output = pos_encoder(x)
		"""

		x = x + self.pe[:x.size(0), :]
		return x

class TransformerModel(nn.Module):
	def __init__(self, config, input_dim):
		super(TransformerModel, self).__init__()
		self.config = config
		self.projection = nn.Linear(input_dim, config.transformer_dim)
		self.pos_encoder = PositionalEncoding(config.transformer_dim, 0.1)
		encoder_layers = TransformerEncoderLayer(config.transformer_dim, config.transformer_head, config.transformer_ff_dim, config.transformer_dropout)
		self.transformer_encoder = TransformerEncoder(encoder_layers, config.transformer_layer)

	def forward(self, src):
		src = self.projection(src) * math.sqrt(self.config.transformer_dim)
		src = self.pos_encoder(src)
		return self.transformer_encoder(src)
