from torch import nn
import torch

class BiAffine(nn.Module):
	"""BiAffine attention layer."""
	def __init__(self, input_dim, output_dim):
		super(BiAffine, self).__init__()
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.U = nn.Parameter(torch.FloatTensor(output_dim, input_dim, input_dim))
		nn.init.xavier_uniform_(self.U)

	def forward(self, R_head, R_dep):
		R_head = R_head.unsqueeze(1)
		R_dep = R_dep.unsqueeze(1)
		Score = R_head @ self.U @ R_dep.transpose(-1, -2)
		return Score.squeeze(1)


class BiaffineScorer(nn.Module):

	def __init__(self, rnn_size, mlp_size, n_label):
		super().__init__()

		mlp_activation = nn.ReLU()

		# The two MLPs that we apply to the RNN output before the biaffine scorer.
		self.arc_head_mlp = nn.Sequential(nn.Linear(rnn_size, mlp_size), mlp_activation)
		self.arc_dep_mlp = nn.Sequential(nn.Linear(rnn_size, mlp_size), mlp_activation)
		self.lab_head_mlp = nn.Sequential(nn.Linear(rnn_size, mlp_size), mlp_activation)
		self.lab_dep_mlp = nn.Sequential(nn.Linear(rnn_size, mlp_size), mlp_activation)

		# Weights for the biaffine part of the model.
		self.arc_biaffine = BiAffine(mlp_size, 1)
		self.lab_biaffine = BiAffine(mlp_size, n_label)

	def forward(self, sentence_repr):
		# MLPs applied to the RNN output: equations 4 and 5 in the paper.
		H_arc_head = self.arc_head_mlp(sentence_repr)
		H_arc_dep = self.arc_dep_mlp(sentence_repr)
		H_lab_head = self.lab_head_mlp(sentence_repr)
		H_lab_dep = self.lab_dep_mlp(sentence_repr)

		arc_score = self.arc_biaffine(H_arc_head, H_arc_dep)  # [batch, sent_lent, sent_lent] (need transpose)
		lab_score = self.lab_biaffine(H_lab_head, H_lab_dep)  # [batch, n_label, sent_lent, sent_lent] (need transpose)
		return arc_score, lab_score
