import torch

def get_input_dim(config, repr_type):
	# Edge scoring module.
	scorer_size = config.rnn_size
	if repr_type == 'uni_bi':
		scorer_size = 4 * config.rnn_size
	if repr_type == 'bi' or repr_type == 'uni':
		scorer_size = 2 * config.rnn_size
	if config.encoder == 'transformer':
		scorer_size = config.transformer_dim
	return scorer_size

def get_repr_from_mode(rnn1_out, rnn2_out, uni_fw, uni_bw, mode):
	if mode == 'uni':
		return rnn1_out
	if mode == 'bi':
		return rnn2_out
	if mode == 'uni_bi':
		return torch.cat([rnn1_out, rnn2_out], dim=2)
	if mode == 'uni_fw':
		return uni_fw
	return uni_bw

def get_dependency_encoder_repr(head_repr_type, dep_repr_type, rnn1_out, rnn2_out, uni_fw, uni_bw):
	head_repr = get_repr_from_mode(rnn1_out, rnn2_out, uni_fw, uni_bw, head_repr_type)
	dep_repr = get_repr_from_mode(rnn1_out, rnn2_out, uni_fw, uni_bw, dep_repr_type)
	return head_repr, dep_repr
