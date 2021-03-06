import math
import torch
from torch.optim.lr_scheduler import LambdaLR
from transformers import AdamW

class LRAdamWPolicy(object):
	def __init__(self, num_warmup_steps, num_training_steps):
		self.num_warmup_steps = num_warmup_steps
		self.num_training_steps = num_training_steps

	def __call__(self, current_step):
		if current_step < self.num_warmup_steps:
			return float(current_step) / float(max(1, self.num_warmup_steps))
		return max(0.0, float(self.num_training_steps - current_step) / float(max(1, self.num_training_steps - self.num_warmup_steps)))

# follow phoNLP
def adamW(model, config, mode):
	if mode is 'teacher':
		base_lr = config.lr_adam
	else:
		base_lr = config.lr_adam_student
	if config.fine_tune:
		params = model.named_parameters()
		no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
		optimizer_grouped_parameters = [
			{"params": [p for n, p in params if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
			{"params": [p for n, p in params if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
		]
		optimizer = AdamW(optimizer_grouped_parameters, betas=config.beta, lr=base_lr, correct_bias=False)
		# To reproduce BertAdam specific behavior set correct_bias=False
	else:
		# standford nlp
		optimizer_grouped_parameters = [p for p in model.parameters() if p.requires_grad]
		optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=base_lr, betas=config.adam_beta, eps=config.adam_eps)

	if config.use_scheduler:
		num_train_optimization_steps = 40 * (8000 / config.batch_size)
		scheduler = LambdaLR(optimizer, LRAdamWPolicy(num_warmup_steps=5, num_training_steps=num_train_optimization_steps), -1)
	else:
		scheduler = None
	return optimizer, scheduler

class LRPolicy(object):
	def __init__(self, config):
		self.config = config

	def __call__(self, step):
		warm_up_multiplier = (min(step, self.config.warm_up_steps) / self.config.warm_up_steps)
		decay_multiplier = 1.0 / (1 + self.config.lr_decay * math.sqrt(step))
		update_factor = warm_up_multiplier * decay_multiplier
		return update_factor

# follow cross-view training
def momentum(model, config, mode):
	if mode == 'teacher':
		base_lr = config.lr_momentum
	else:
		base_lr = config.student_lr_momentum
	params = model.parameters()
	optimizer = torch.optim.SGD(params, lr=base_lr, momentum=config.momentum)
	if config.use_scheduler:
		scheduler = LambdaLR(optimizer, lr_lambda=LRPolicy(config))
	else:
		scheduler = None
	return optimizer, scheduler

