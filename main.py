import numpy as np
import random
import torch

from config.default_config import Config
from utils import utils
from models.deep_biaffine import DeepBiaffine
from preprocess import dataset

def init_random(config, seed):
	torch.manual_seed(seed)
	np.random.seed(seed)
	random.seed(seed)
	if config.use_cuda:
		torch.cuda.manual_seed(seed)

def main():
	config = Config()
	init_random(config, config.seed)
	if config.mode == 'train':
		train(config)
	elif config.mode == 'evaluate':
		evaluate(config)
	else:
		annotate(config)

def train(config):
	# get data
	utils.ensure_dir(config.save_folder)
	train_loader = dataset.get_data_loader(config.train_file, config)
	dev_loader = dataset.get_data_loader(config.dev_file, config)
	test_loader = dataset.get_data_loader(config.test_file, config)

	model = DeepBiaffine()

	for epoch in range(config.epoch):
		for i, data in enumerate(train_loader, 0):
			if i == 0:
				utils.log(data)



def evaluate(config):
	pass

def annotate(config):
	pass


if __name__ == '__main__':
	main()
