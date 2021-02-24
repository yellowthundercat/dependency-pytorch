from collections import defaultdict, Counter
import time
import numpy as np
import random
import torch
import torchtext
import matplotlib.pyplot as plt

from config.default_config import Config
from utils import utils
from models.edge_parser import EdgeFactoredParser
from models.deep_biaffine import DeepBiaffine
from preprocess import dataset


class DependencyParser:
	def __init__(self, lower=False):
		pad = '<pad>'
		self.WORD = torchtext.data.Field(init_token=pad, pad_token=pad, sequential=True, lower=lower, batch_first=True)
		self.POS = torchtext.data.Field(init_token=pad, pad_token=pad, sequential=True, batch_first=True)
		self.HEAD = torchtext.data.Field(init_token=0, pad_token=0, use_vocab=False, sequential=True, batch_first=True)
		# self.DEPENDENCY_LABEL = torchtext.data.Field(init_token=0, pad_token=0, use_vocab=False, sequential=True, batch_first=True)
		self.fields = [('words', self.WORD), ('postags', self.POS), ('heads', self.HEAD)]
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	def train(self, config):
		# get data
		utils.ensure_dir(config.save_folder)
		train_examples = dataset.read_data(config.train_file, config, self.fields)
		dev_examples = dataset.read_data(config.dev_file, config, self.fields)
		# test_examples = dataset.read_data(config.test_file, config, self.fields)

		print('We are training word embeddings from scratch.')
		self.WORD.build_vocab(train_examples, max_size=10000)
		self.POS.build_vocab(train_examples)

		self.model = EdgeFactoredParser(self.fields, word_emb_dim=300, pos_emb_dim=32, rnn_size=256, rnn_depth=3,
																		mlp_size=256, update_pretrained=False)
		self.model.to(self.device)
		train_iterator = torchtext.data.BucketIterator(
			train_examples,
			device=self.device,
			batch_size=config.batch_size,
			sort_key=lambda x: len(x.words),
			repeat=False,
			train=True,
			sort=True)
		val_iterator = torchtext.data.BucketIterator(
			dev_examples,
			device=self.device,
			batch_size=config.batch_size,
			sort_key=lambda x: len(x.words),
			repeat=False,
			train=True,
			sort=True)
		train_batches = list(train_iterator)
		val_batches = list(val_iterator)

		optimizer = torch.optim.Adam(self.model.parameters(), betas=(0.9, 0.9), lr=0.005, weight_decay=1e-5)
		history = defaultdict(list)

		for i in range(1, config.epoch + 1):
			t0 = time.time()
			stats = Counter()

			self.model.train()
			for batch in train_batches:
				loss = self.model(batch.words, batch.postags, batch.heads)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
				stats['train_loss'] += loss.item()

			train_loss = stats['train_loss'] / len(train_batches)
			history['train_loss'].append(train_loss)

			self.model.eval()
			with torch.no_grad():
				for batch in val_batches:
					loss, n_err, n_tokens = self.model(batch.words, batch.postags, batch.heads, evaluate=True)
					stats['val_loss'] += loss.item()
					stats['val_n_tokens'] += n_tokens
					stats['val_n_err'] += n_err

			val_loss = stats['val_loss'] / len(val_batches)
			uas = (stats['val_n_tokens'] - stats['val_n_err']) / stats['val_n_tokens']
			history['val_loss'].append(val_loss)
			history['uas'].append(uas)

			t1 = time.time()
			print(f'Epoch {i}: train loss = {train_loss:.4f}, val loss = {val_loss:.4f}, UAS = {uas:.4f}, time = {t1 - t0:.4f}')

		plt.plot(history['train_loss'])
		plt.plot(history['val_loss'])
		plt.plot(history['uas'])
		plt.legend(['training loss', 'validation loss', 'UAS'])

	def evaluate(self, config):
		pass

	def annotate(self, config):
		pass


def main():
	config = Config()
	parser = DependencyParser()
	if config.mode == 'train':
		parser.train(config)
	elif config.mode == 'evaluate':
		parser.evaluate(config)
	else:
		parser.annotate(config)


if __name__ == '__main__':
	main()
