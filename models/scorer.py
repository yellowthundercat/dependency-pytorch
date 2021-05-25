from torch import nn
import torch.nn.functional as F
import torch
from models.deep_biaffine import DeepBiaffineScorer

class BiAffine(nn.Module):
	"""BiAffine attention layer."""
	def __init__(self, config, rnn_size, mlp_size, output_dim, mlp_dropout):
		super(BiAffine, self).__init__()
		mlp_activation = nn.ReLU()
		self.head_mlp = nn.Sequential(nn.Linear(rnn_size, mlp_size), mlp_activation)
		self.dep_mlp = nn.Sequential(nn.Linear(rnn_size, mlp_size), mlp_activation)
		self.head_dropout = nn.Dropout(p=mlp_dropout)
		self.dep_dropout = nn.Dropout(p=mlp_dropout)

		self.input_dim = mlp_size
		self.output_dim = output_dim
		self.U = nn.Parameter(torch.FloatTensor(output_dim, mlp_size+1, mlp_size+1))
		nn.init.xavier_uniform_(self.U)

	def forward(self, sentence_repr):
		R_head = self.head_mlp(sentence_repr)
		R_dep = self.dep_mlp(sentence_repr)
		# add bias
		R_head = torch.cat([R_head, R_head.new_ones(*R_head.size()[:-1], 1)], len(R_head.size()) - 1)
		R_dep = torch.cat([R_dep, R_dep.new_ones(*R_dep.size()[:-1], 1)], len(R_dep.size()) - 1)

		R_head = self.head_dropout(R_head)
		R_dep = self.dep_dropout(R_dep)

		# calculate
		R_head = R_head.unsqueeze(1)
		R_dep = R_dep.unsqueeze(1)
		Score = R_head @ self.U @ R_dep.transpose(-1, -2)
		return Score.squeeze(1)


class BiaffineScorer(nn.Module):

	def __init__(self, config, rnn_size, n_label, dropout):
		super().__init__()
		self.config = config
		# Weights for the biaffine part of the model.
		# self.arc_biaffine = BiAffine(config, rnn_size, config.arc_mlp_size, 1, dropout)
		# self.lab_biaffine = BiAffine(config, rnn_size, config.lab_mlp_size, n_label, dropout)
		self.arc_biaffine = DeepBiaffineScorer(rnn_size, rnn_size, config.arc_mlp_size, 1, dropout=dropout)
		self.lab_biaffine = DeepBiaffineScorer(rnn_size, rnn_size, config.lab_mlp_size, n_label, dropout=dropout)

		self.linear_order = DeepBiaffineScorer(rnn_size, rnn_size, config.arc_mlp_size, 1, dropout=dropout)
		self.distance = DeepBiaffineScorer(rnn_size, rnn_size, config.arc_mlp_size, 1, dropout=dropout)

	def forward(self, head_repr, dep_repr):
		arc_score = self.arc_biaffine(head_repr, dep_repr).squeeze(-1)  # [batch, sent_lent, sent_lent] (need transpose later in parser)
		lab_score = self.lab_biaffine(head_repr, dep_repr).transpose(-1, -3)  # [batch, n_label, sent_lent, sent_lent] (need transpose later in parser)

		head_offset = torch.arange(head_repr.size(1), device=arc_score.device).view(1, 1, -1).expand(
			head_repr.size(0), -1, -1) - torch.arange(head_repr.size(1), device=arc_score.device).view(1, -1, 1).expand(
			head_repr.size(0), -1, -1)

		lin_scores = self.linear_order(head_repr, dep_repr).squeeze(3)
		arc_score += F.logsigmoid(lin_scores * torch.sign(head_offset).float()).detach()

		dist_scores = self.distance(head_repr, dep_repr).squeeze(3)
		dist_pred = 1 + F.softplus(dist_scores)
		dist_target = torch.abs(head_offset)
		dist_kld = -torch.log((dist_target.float() - dist_pred) ** 2 / 2 + 1)
		arc_score += dist_kld.detach()

		return arc_score, lab_score, lin_scores, dist_kld, head_offset
