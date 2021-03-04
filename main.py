from collections import defaultdict, Counter
import os
import time
import numpy as np
import random
import torch
import torchtext

from config.default_config import Config
from utils import utils
from models.edge_parser import EdgeFactoredParser
from models.deep_biaffine import DeepBiaffine
from preprocess import dataset


class DependencyParser:
	def __init__(self, config):
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.config = config
		self.corpus = dataset.Corpus(config)
		if os.path.exists(config.model_file):
			print('We will continue training')
			self.model = torch.load(config.model_file)
		else:
			print('We will train model from scratch')
			self.model = EdgeFactoredParser(len(self.corpus.vocab.t2i), config,
																			word_emb_dim=config.word_emb_dim, pos_emb_dim=config.pos_emb_dim,
																			rnn_size=config.rnn_size, rnn_depth=config.rnn_depth,
																			mlp_size=config.mlp_size, update_pretrained=False)
		self.model.to(self.device)

	def train(self):
		print('start training')
		optimizer = torch.optim.Adam(self.model.parameters(), betas=(0.9, 0.9), lr=0.005, weight_decay=1e-5)
		history = defaultdict(list)

		best_uas = 0.
		best_epoch = 0

		for epoch_index in range(1, self.config.epoch + 1):
			t0 = time.time()
			stats = Counter()

			train_batches = self.corpus.train.batches(self.config.batch_size, length_ordered=self.config.length_ordered)
			self.model.train()
			for batch in train_batches:
				words, tags, heads, labels, masks = batch
				loss = self.model(words, tags, heads, labels, masks)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
				stats['train_loss'] += loss.item()

			train_loss = stats['train_loss'] / len(train_batches)
			history['train_loss'].append(train_loss)

			self.model.eval()
			dev_batches = self.corpus.dev.batches(self.config.batch_size, length_ordered=True)
			dev_batch_length = 0
			with torch.no_grad():
				for batch in dev_batches:
					dev_batch_length += 1
					words, tags, heads, labels, masks = batch
					loss, n_err, n_tokens = self.model(words, tags, heads, labels, masks, evaluate=True)
					stats['val_loss'] += loss.item()
					stats['val_n_tokens'] += n_tokens
					stats['val_n_err'] += n_err

			val_loss = stats['val_loss'] / dev_batch_length
			uas = (stats['val_n_tokens'] - stats['val_n_err']) / stats['val_n_tokens']
			history['val_loss'].append(val_loss)
			history['uas'].append(uas)

			if uas > best_uas:
				torch.save(self.model, self.config.model_file)
				best_uas = uas
				best_epoch = epoch_index

			t1 = time.time()
			print(f'Epoch {epoch_index}: train loss = {train_loss:.4f}, val loss = {val_loss:.4f}, UAS = {uas:.4f}, time = {t1 - t0:.4f}')

		utils.show_history_graph(history)
		print('finish training')
		print('best uas:', best_uas)
		print('best epoch', best_epoch)

	def evaluate(self):
		print('evaluating')


	def annotate(self):
		print('parsing')


def main():
	# load config
	config = Config()
	utils.ensure_dir(config.save_folder)
	if os.path.exists(config.config_file):
		config = torch.load(config.config_file)

	parser = DependencyParser(config)

	if config.mode == 'train':
		parser.train()
	elif config.mode == 'evaluate':
		parser.evaluate()
	else:
		parser.annotate()


if __name__ == '__main__':
	main()
