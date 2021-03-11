from collections import defaultdict, Counter
import os
import time
import numpy as np
import random
import torch
import torchtext

from config.default_config import Config
from utils import utils
from models.parser import Parser
from models.deep_biaffine import DeepBiaffine
from preprocess import dataset


class DependencyParser:
	def __init__(self, config):
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.config = config
		if os.path.exists(config.corpus_file):
			print('load preprocessed corpus')
			self.corpus = torch.load(config.corpus_file)
		else:
			print('preprocess corpus')
			self.corpus = dataset.Corpus(config, self.device)
			torch.save(self.corpus, config.corpus_file)
		if os.path.exists(config.model_file):
			print('We will continue training')
			self.model = torch.load(config.model_file)
		else:
			print('We will train model from scratch')
			self.model = Parser(len(self.corpus.vocab.t2i), len(self.corpus.vocab.l2i), config,
																			word_emb_dim=config.word_emb_dim, pos_emb_dim=config.pos_emb_dim,
																			rnn_size=config.rnn_size, rnn_depth=config.rnn_depth,
																			mlp_size=config.mlp_size, update_pretrained=False)
		self.model.to(self.device)

	def train(self):
		print('start training')
		optimizer = torch.optim.Adam(self.model.parameters(), betas=(0.9, 0.9), lr=0.005, weight_decay=1e-5)
		history = defaultdict(list)

		best_uas = 0.
		best_las = 0.
		best_epoch = 0

		for epoch_index in range(1, self.config.epoch + 1):
			t0 = time.time()
			stats = Counter()

			train_batches = self.corpus.train.batches(self.config.batch_size, length_ordered=self.config.length_ordered)
			train_batch_length = 0
			self.model.train()
			for batch in train_batches:
				train_batch_length += 1
				words, tags, heads, labels, masks, lengths, origin_words = batch
				loss = self.model(words, tags, heads, labels, masks)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
				stats['train_loss'] += loss.item()

			train_loss = stats['train_loss'] / train_batch_length
			history['train_loss'].append(train_loss)

			self.model.eval()
			dev_batches = self.corpus.dev.batches(self.config.batch_size, length_ordered=False, origin_ordered=True)
			dev_batch_length = 0
			dev_word_list = []
			dev_length_list = []
			dev_head_list = []
			dev_lab_list = []
			with torch.no_grad():
				for batch in dev_batches:
					dev_batch_length += 1
					words, tags, heads, labels, masks, lengths, origin_words = batch
					loss, head_list, lab_list = self.model.predict_batch_with_loss(words, tags, heads, labels, masks, lengths)
					stats['val_loss'] += loss.item()
					dev_head_list += head_list
					dev_lab_list += lab_list
					dev_word_list += origin_words
					dev_length_list += lengths

			utils.write_conll(self.corpus.vocab, dev_word_list, dev_head_list, dev_lab_list, dev_length_list, self.config.parsing_file)
			val_loss = stats['val_loss'] / dev_batch_length
			uas, las = utils.ud_scores(self.config.dev_file, self.config.parsing_file)
			history['val_loss'].append(val_loss)
			history['uas'].append(uas)
			history['las'].append(las)

			if las > best_las:
				torch.save(self.model, self.config.model_file)
				best_las = las
				best_epoch = epoch_index

			if uas > best_las:
				best_uas = uas

			t1 = time.time()
			print(f'Epoch {epoch_index}: train loss = {train_loss:.4f}, val loss = {val_loss:.4f}, UAS = {uas:.4f}, LAS = {las:.4f}, time = {t1 - t0:.4f}')

		torch.save(self.config, self.config.config_file)
		# utils.show_history_graph(history)
		print('finish training')
		print('best uas:', best_uas)
		print('best las:', best_las)
		print('best epoch las', best_epoch)
		print('-'*20)
		self.evaluate()

	def evaluate(self):
		print('evaluating')
		self.model.eval()
		test_batches = self.corpus.test.batches(self.config.batch_size, length_ordered=False, origin_ordered=True)
		test_batch_length = 0
		test_word_list = []
		test_length_list = []
		test_head_list = []
		test_lab_list = []
		with torch.no_grad():
			for batch in test_batches:
				test_batch_length += 1
				words, tags, heads, labels, masks, lengths, origin_words = batch
				head_list, lab_list = self.model.predict_batch(words, tags, lengths)
				test_head_list += head_list
				test_lab_list += lab_list
				test_word_list += origin_words
				test_length_list += lengths
			utils.write_conll(self.corpus.vocab, test_word_list, test_head_list, test_lab_list, test_length_list,
												self.config.parsing_file)
			uas, las = utils.ud_scores(self.config.test_file, self.config.parsing_file)
			print(f'Evaluating Result: UAS = {uas:.4f}, LAS = {las:.4}')

	def annotate(self):
		print('parsing')
		input_file = open(self.config.annotate_file, encoding='utf-8')



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
