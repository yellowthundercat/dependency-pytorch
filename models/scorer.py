from torch import nn
import torch.nn.functional as F
import torch
from models.deep_biaffine import DeepBiaffineScorer


class BiaffineScorer(nn.Module):

	def __init__(self, config, rnn_size, n_label, dropout):
		super().__init__()
		self.config = config
		self.n_label = n_label

		# Weights for the biaffine part of the model.
		self.arc_biaffine = DeepBiaffineScorer(rnn_size, rnn_size, config.arc_mlp_size, 1, dropout=dropout)
		self.lab_biaffine = DeepBiaffineScorer(rnn_size, rnn_size, config.lab_mlp_size, n_label, dropout=dropout)

		if config.use_linearization:
			self.linear_order = DeepBiaffineScorer(rnn_size, rnn_size, config.arc_mlp_size, 1, dropout=dropout)
		if config.use_distance:
			self.distance = DeepBiaffineScorer(rnn_size, rnn_size, config.arc_mlp_size, 1, dropout=dropout)

		# criterion
		self.crit = nn.CrossEntropyLoss(ignore_index=-1, reduction='sum')  # ignore padding
		self.drop = nn.Dropout(p=dropout)

	def forward(self, head_repr, dep_repr, heads, labels, masks):
		arc_score = self.arc_biaffine(self.drop(head_repr), self.drop(dep_repr)).squeeze(3)
		lab_score = self.lab_biaffine(self.drop(head_repr), self.drop(dep_repr))

		if self.config.use_distance or self.config.use_linearization:
			head_offset = torch.arange(head_repr.size(1), device=arc_score.device).view(1, 1, -1).expand(
				head_repr.size(0), -1, -1) - torch.arange(head_repr.size(1), device=arc_score.device).view(1, -1, 1).expand(
				head_repr.size(0), -1, -1)

		if self.config.use_linearization:
			lin_scores = self.linear_order(self.drop(head_repr), self.drop(dep_repr)).squeeze(3)
			arc_score += F.logsigmoid(lin_scores * torch.sign(head_offset).float()).detach()

		if self.config.use_distance:
			dist_scores = self.distance(self.drop(head_repr), self.drop(dep_repr)).squeeze(3)
			dist_pred = 1 + F.softplus(dist_scores)
			dist_target = torch.abs(head_offset)
			dist_kld = -torch.log((dist_target.float() - dist_pred) ** 2 / 2 + 1)
			arc_score += dist_kld.detach()

		heads = heads[:, 1:]  # adapt with parser from standford
		labels = labels[:, 1:]
		diag = torch.eye(heads.size(-1) + 1, dtype=torch.bool, device=heads.device).unsqueeze(0)
		arc_score.masked_fill_(diag, -float('inf'))

		preds = []

		unlabeled_scores = arc_score[:, 1:, :]  # exclude attachment for the root symbol
		unlabeled_scores = unlabeled_scores.masked_fill(masks.unsqueeze(1), -float('inf'))
		unlabeled_target = heads.masked_fill(masks[:, 1:], -1)
		loss = self.crit(unlabeled_scores.contiguous().view(-1, unlabeled_scores.size(2)), unlabeled_target.view(-1))

		deprel_scores = lab_score[:, 1:]  # exclude attachment for the root symbol
		# deprel_scores = deprel_scores.masked_select(goldmask.unsqueeze(3)).view(-1, len(self.vocab['deprel']))
		deprel_scores = torch.gather(deprel_scores, 2, heads.unsqueeze(2).unsqueeze(3).expand(-1, -1, -1, self.n_label)).view(-1, self.n_label)
		deprel_target = labels.masked_fill(masks[:, 1:], -1)
		loss += self.crit(deprel_scores.contiguous(), deprel_target.view(-1))

		if self.config.use_linearization:
			# lin_scores = lin_scores[:, 1:].masked_select(goldmask)
			lin_scores = torch.gather(lin_scores[:, 1:], 2, heads.unsqueeze(2)).view(-1)
			lin_scores = torch.cat([-lin_scores.unsqueeze(1) / 2, lin_scores.unsqueeze(1) / 2], 1)
			# lin_target = (head_offset[:, 1:] > 0).long().masked_select(goldmask)
			lin_target = torch.gather((head_offset[:, 1:] > 0).long(), 2, heads.unsqueeze(2))
			loss += self.crit(lin_scores.contiguous(), lin_target.view(-1))

		if self.config.use_distance:
			# dist_kld = dist_kld[:, 1:].masked_select(goldmask)
			dist_kld = torch.gather(dist_kld[:, 1:], 2, heads.unsqueeze(2))
			loss -= dist_kld.sum()

		loss /= masks.sum()  # number of words

		preds.append(F.log_softmax(arc_score, 2).detach().cpu().numpy())
		preds.append(lab_score.max(3)[1].detach().cpu().numpy())

		return loss, preds
