from torch import nn

class BiaffineScorer(nn.Module):

	def __init__(self, rnn_size, mlp_size):
		super().__init__()

		mlp_activation = nn.ReLU()

		# The two MLPs that we apply to the RNN output before the biaffine scorer.
		self.head_mlp = nn.Sequential(nn.Linear(rnn_size, mlp_size), mlp_activation)
		self.dep_mlp = nn.Sequential(nn.Linear(rnn_size, mlp_size), mlp_activation)

		# Weights for the biaffine part of the model.
		self.W_arc = nn.Linear(mlp_size, mlp_size, bias=False)
		self.b_arc = nn.Linear(mlp_size, 1, bias=False)

	def forward(self, sentence_repr):
		# MLPs applied to the RNN output: equations 4 and 5 in the paper.
		H_arc_head = self.head_mlp(sentence_repr)
		H_arc_dep = self.dep_mlp(sentence_repr)

		# Computing the edge scores for all edges using the biaffine model.
		# This corresponds to equation 9 in the paper. For readability we implement this
		# in a step-by-step fashion.
		Hh_W = self.W_arc(H_arc_head)
		Hh_W_Ha = H_arc_dep.matmul(Hh_W.transpose(1, 2))
		Hh_b = self.b_arc(H_arc_head).transpose(1, 2)
		return Hh_W_Ha + Hh_b