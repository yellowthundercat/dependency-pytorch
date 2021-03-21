from collections import defaultdict, Counter
import os
import time
import numpy as np
import random
import torch
import torchtext
from models.encoder import RNNEncoder

from config.default_config import Config
from utils import utils
from models.parser import Parser
from models.deep_biaffine import DeepBiaffine
from preprocess import dataset
from utils.find_error_example import write_file_error_example

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
		self.unlabel_corpus = dataset.Unlabel_Corpus(config, self.device, self.corpus.vocab)
		if os.path.exists(config.model_file):
			print('We will continue training')
			all_model = torch.load(config.model_file)
			self.encoder = all_model['encoder']
			self.model = all_model['model']
			self.model.encoder = self.encoder
			self.model_students = all_model['model_students']
			for student_model in self.model_students:
				student_model.encoder = self.encoder
			self.optimizer = all_model['optimizer']
			self.optimizer_students = all_model['optimizer_students']
			self.saving_epoch = all_model['epoch']
			self.best_uas = all_model['uas']
			self.best_las = all_model['las']
		else:
			print('We will train model from scratch')
			self.encoder = RNNEncoder(config, len(self.corpus.vocab.t2i))
			self.model = Parser(self.encoder, len(self.corpus.vocab.l2i), config, 'uni_bi', 'uni_bi')
			self.model_students = [
				Parser(self.encoder, len(self.corpus.vocab.l2i), config, 'uni_bi', 'uni_bi'),
				Parser(self.encoder, len(self.corpus.vocab.l2i), config, 'uni_fw', 'uni_fw'),
				Parser(self.encoder, len(self.corpus.vocab.l2i), config, 'uni_fw', 'uni_bw'),
				Parser(self.encoder, len(self.corpus.vocab.l2i), config, 'uni_bw', 'uni_fw'),
				Parser(self.encoder, len(self.corpus.vocab.l2i), config, 'uni_bw', 'uni_bw')
			]
			self.optimizer = torch.optim.Adam(self.model.parameters(), betas=(0.9, 0.9), lr=0.005, weight_decay=1e-5)
			self.optimizer_students = [torch.optim.Adam(model_student.parameters(), betas=(0.9, 0.9), lr=0.005, weight_decay=1e-5)
																for model_student in self.model_students]
			self.saving_epoch = 0
			self.best_las = self.best_uas = 0
		self.model.to(self.device)
		for model_student in self.model_students:
			model_student.to(self.device)

	def save_all_model(self):
		all_model = {
			'encoder': self.encoder,
			'model': self.model,
			'model_students': self.model_students,
			'optimizer': self.optimizer,
			'optimizer_students': self.optimizer_students,
			'epoch': self.saving_epoch,
			'uas': self.best_uas,
			'las': self.best_las
		}
		torch.save(all_model, self.config.model_file)

	def check_dev(self):
		stats = Counter()
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

		utils.write_conll(self.corpus.vocab, dev_word_list, dev_head_list, dev_lab_list, dev_length_list,
											self.config.parsing_file)
		val_loss = stats['val_loss'] / dev_batch_length
		uas, las = utils.ud_scores(self.config.dev_file, self.config.parsing_file)
		return val_loss, uas, las


	def train(self):
		print('start training')

		history = defaultdict(list)

		for epoch_index in range(self.saving_epoch+1, self.config.epoch + 1):
			t0 = time.time()
			stats = Counter()

			# train teacher model
			train_batches = self.corpus.train.batches(self.config.batch_size, length_ordered=self.config.length_ordered)
			train_batch_length = 0
			self.model.train()
			for batch in train_batches:
				train_batch_length += 1
				words, tags, heads, labels, masks, lengths, origin_words = batch
				loss = self.model(words, tags, heads, labels, masks)
				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()
				stats['train_loss'] += loss.item()

			train_loss = stats['train_loss'] / train_batch_length
			history['train_loss'].append(train_loss)

			# train students model
			self.model.eval()
			# get new predict head, label
			predict_heads = []
			predict_labels = []
			unlabel_batches = self.unlabel_corpus.dataset.batches(self.config.batch_size, length_ordered=False, origin_ordered=True)
			for batch in unlabel_batches:
				words, tags, heads, labels, masks, lengths, origin_words = batch
				head_list, lab_list = self.model.predict_batch(words, tags, lengths)
				predict_heads += [heads.tolist() for heads in head_list]
				predict_labels += [lab.tolist() for lab in lab_list]
			self.unlabel_corpus.dataset.heads = predict_heads
			self.unlabel_corpus.dataset.labels = predict_labels

			for student_model, studet_optimizer in zip(self.model_students, self.optimizer_students):
				student_model.train()
				unlabel_batches = self.unlabel_corpus.dataset.batches(self.config.batch_size, length_ordered=self.config.length_ordered)
				for batch in unlabel_batches:
					words, tags, heads, labels, masks, lengths, origin_words = batch
					loss = self.model(words, tags, heads, labels, masks)
					studet_optimizer.zero_grad()
					loss.backward()
					studet_optimizer.step()

			# check dev
			val_loss, uas, las = self.check_dev()
			history['val_loss'].append(val_loss)
			history['uas'].append(uas)
			history['las'].append(las)

			if las + uas > self.best_las + self.best_uas:
				print('save new best model')
				self.best_las = las
				self.best_uas = uas
				self.saving_epoch = epoch_index
				self.save_all_model()

			t1 = time.time()
			print(f'Epoch {epoch_index}: train loss = {train_loss:.4f}, val loss = {val_loss:.4f}, UAS = {uas:.4f}, LAS = {las:.4f}, time = {t1 - t0:.4f}')

		# torch.save(self.config, self.config.config_file)
		# utils.show_history_graph(history)
		print('finish training')
		print('best uas:', self.best_uas)
		print('best las:', self.best_las)
		print('best epoch', self.saving_epoch)
		print('-'*20)
		self.evaluate()

	def evaluate(self):
		all_model = torch.load(self.config.model_file)
		self.model = all_model['model']
		self.model.to(self.device)
		print('evaluating')
		self.model.eval()
		test_batches = self.corpus.test.batches(self.config.batch_size, length_ordered=False)
		test_batch_length = 0
		test_word_list = []
		test_length_list = []
		test_head_list = []
		test_lab_list = []
		gold_head_list = []
		gold_lab_list = []
		with torch.no_grad():
			for batch in test_batches:
				test_batch_length += 1
				words, tags, heads, labels, masks, lengths, origin_words = batch
				head_list, lab_list = self.model.predict_batch(words, tags, lengths)
				gold_head_list += [head.data.numpy()[:lent] for head, lent in zip(heads.cpu(), lengths)]
				gold_lab_list += [lab.data.numpy()[:lent] for lab, lent in zip(labels.cpu(), lengths)]
				test_head_list += head_list
				test_lab_list += lab_list
				test_word_list += origin_words
				test_length_list += lengths
			utils.write_conll(self.corpus.vocab, test_word_list, test_head_list, test_lab_list, test_length_list,
												self.config.parsing_file)
			write_file_error_example(self.config, self.corpus.vocab, test_word_list, test_head_list, gold_head_list, test_lab_list,
															 gold_lab_list, test_length_list)
			uas, las = utils.ud_scores(self.config.test_file, self.config.parsing_file)
			print(f'Evaluating Result: UAS = {uas:.4f}, LAS = {las:.4}')

	def annotate(self):
		print('parsing')
		input_file = open(self.config.annotate_file, encoding='utf-8')
		sentence = input_file.read()



def main():
	# load config
	config = Config()
	utils.ensure_dir(config.save_folder)
	utils.ensure_dir(config.unlabel_embedding_folder)
	if os.path.exists(config.config_file):
		config = torch.load(config.config_file)

	t0 = time.time()
	parser = DependencyParser(config)
	t1 = time.time()
	print(f'init time = {t1 - t0:.4f}')

	if config.mode == 'train':
		parser.train()
	elif config.mode == 'evaluate':
		parser.evaluate()
	else:
		parser.annotate()


if __name__ == '__main__':
	main()
