import math
import torch
from torch.optim.lr_scheduler import LambdaLR

def adam(params, config):
	return torch.optim.Adam(params, betas=(0.9, 0.9), lr=0.005, weight_decay=1e-5)

def momentum(params, config):
	return torch.optim.SGD(params, lr=config.lr_momentum, momentum=config.momentum)

class LRPolicy(object):
	def __init__(self, config):
		self.config = config

	def __call__(self, step):
		warm_up_multiplier = (min(step, self.config.warm_up_steps) / self.config.warm_up_steps)
		decay_multiplier = 1.0 / (1 + self.config.lr_decay * math.sqrt(step))
		update_factor = warm_up_multiplier * decay_multiplier
		return update_factor

def momentum_scheduler(optimizer, config):
	scheduler = LambdaLR(optimizer, lr_lambda=LRPolicy(config))
	return scheduler
