from collections import defaultdict, Counter
import os
import time
import random
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import nltk
from graphviz import Source

from config.default_config import Config
from utils import utils
from preprocess import dataset, sentence_level
from utils.find_error_example import write_file_error_example
from utils import utils_train
from torch import nn, optim
from preprocess.char import ROOT_LABEL

class DependencyParser:
	def __init__(self, config):
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.config = config

		# word embedding
		print('preprocess corpus')
		self.tokenizer = self.phobert = None
		if config.use_phobert:
			self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
			self.phobert = AutoModel.from_pretrained("vinai/phobert-base", output_hidden_states=True).to(self.device)
		self.corpus = dataset.Corpus(config, self.device, self.tokenizer, self.phobert)
		if config.cross_view and config.mode == 'train':
			print('prepare unlabel data')
			self.unlabel_corpus = dataset.Unlabel_Corpus(config, self.device, self.corpus.vocab, self.tokenizer, self.phobert)
		print('total vocab word', len(self.corpus.vocab.w2i))
		print('total vocab character', len(self.corpus.vocab.c2i))
		print('total vocab postag', len(self.corpus.vocab.t2i))
		self.config.pos_label_dim = len(self.corpus.vocab.t2i)

		self.best_model = None


	def internal_train_student(self, words, index_ids, last_index_position, tags, chars, heads, labels, masks, lengths):
		total_loss = 0
		for student_model, student_optimizer, student_scheduler in zip(self.model_students, self.optimizer_students, self.scheduler_students):
			student_optimizer.zero_grad()
			student_model.train()
			student_model.encoder.mode = 'student'
			loss = student_model(words, index_ids, last_index_position, tags, chars, heads, labels, masks, lengths)
			loss.backward()
			if self.config.grad_clip_adam and not self.config.use_momentum:
				torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)
			student_optimizer.step()
			if self.config.use_scheduler:
				student_scheduler.step()
			total_loss += loss.item()
		return total_loss

	def train_gold_student(self, gold_batch):
		words, index_ids, last_index_position, tags, heads, labels, masks, lengths, origin_words, chars, new_order = gold_batch
		return self.internal_train_student(words, index_ids, last_index_position, tags, chars, heads, labels, masks, lengths)

	def train_student(self, unlabel_batch):
		# use teacher to predict
		self.model.eval()
		self.model.encoder.mode = 'teacher'
		words, index_ids, last_index_position, tags, heads, labels, masks, lengths, origin_words, chars, new_order = unlabel_batch
		head_list, lab_list = self.model.predict_batch(words, index_ids, last_index_position, tags, chars, heads, labels, lengths, masks)
		predict_heads = [[0] + one_head.tolist() for one_head in head_list]
		predict_labels = [[self.corpus.vocab.l2i[ROOT_LABEL]] + one_lab for one_lab in lab_list]
		predict_heads = dataset.pad(predict_heads)
		predict_labels = dataset.pad(predict_labels)
		return self.internal_train_student(words, index_ids, last_index_position, tags, chars, predict_heads, predict_labels, masks, lengths)

	def train_teacher(self, train_batch):
		words, index_ids, last_index_position, tags, heads, labels, masks, lengths, origin_words, chars, new_order = train_batch
		self.optimizer.zero_grad()
		self.model.train()
		self.model.encoder.mode = 'teacher'
		loss = self.model(words, index_ids, last_index_position, tags, chars, heads, labels, masks, lengths)
		loss.backward()
		if self.config.grad_clip_adam and not self.config.use_momentum:
			torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
		self.optimizer.step()
		if self.config.use_scheduler:
			self.scheduler.step()
		return loss.item()

	def get_train_batch(self, batches, is_label=True):
		try:
			batch = next(batches)
		except StopIteration:
			if is_label is False:
				batches = self.unlabel_corpus.dataset.batches(self.config.batch_size, length_ordered=self.config.length_ordered)
			else:
				batches = self.corpus.train.batches(self.config.batch_size, length_ordered=self.config.length_ordered)
			batch = next(batches)
		return batch, batches

	def train(self):
		# model
		is_best = os.path.exists(self.config.model_file)
		is_last = os.path.exists(self.config.last_model_file)
		if (is_last or is_best) and self.config.continue_train:
			print('We will continue training')
			if is_last:
				all_model = torch.load(self.config.last_model_file, map_location=self.device)
			else:
				all_model = torch.load(self.config.model_file, map_location=self.device)
			utils_train.load_model(self, all_model, self.config)
		else:
			print('We will train model from scratch')
			utils_train.init_model(self, self.config)
		self.encoder.device = self.device
		self.model.to(self.device)
		if self.config.cross_view:
			for model_student in self.model_students:
				model_student.to(self.device)

		print('start training')

		history = defaultdict(list)
		total_teacher_loss = total_student_loss = 0
		count_teacher = count_student = 0
		train_batches = self.corpus.train.batches(self.config.batch_size, length_ordered=self.config.length_ordered)
		if self.config.cross_view:
			unlabel_batches = self.unlabel_corpus.dataset.batches(self.config.batch_size, length_ordered=self.config.length_ordered)
		# odd for teacher, even for student
		t0 = time.time()
		for global_step in range(self.saving_step+1, self.config.max_step+1):
			# train
			if global_step <= self.config.teacher_only_step or global_step % 2 == 1 or self.config.cross_view is False:
				# train teacher
				train_batch, train_batches = self.get_train_batch(train_batches, is_label=True)
				loss = self.train_teacher(train_batch)
				total_teacher_loss += loss
				count_teacher += 1
			else:
				# train student
				if (global_step // 2) % self.config.gold_student_step == 0:
					train_batch, train_batches = self.get_train_batch(train_batches, is_label=True)
					total_student_loss += self.train_gold_student(train_batch)
				else:
					unlabel_batch, unlabel_batches = self.get_train_batch(unlabel_batches, is_label=False)
					total_student_loss += self.train_student(unlabel_batch)
				count_student += 5

			# switch optimizer
			if global_step - self.saving_step > self.config.max_waiting_adam and not self.config.use_momentum and not self.using_amsgrad:
				print('Switching to AMSGrad')
				self.using_amsgrad = True
				self.optimizer = optim.Adam(self.model.parameters(), amsgrad=True, lr=self.config.lr_adam, betas=self.config.adam_beta, eps=1e-6)
				if self.config.cross_view:
					for i_m in range(len(self.model_students)):
						self.optimizer_students[i_m] = optim.Adam(self.model_students[i_m].parameters(), amsgrad=True, lr=self.config.lr_adam, betas=self.config.adam_beta, eps=1e-6)

			# print result
			if global_step % self.config.print_step == 0 or global_step == self.config.max_step:
				t1 = time.time()
				teacher_loss = total_teacher_loss/max(1, count_teacher)
				student_loss = total_student_loss/max(1, count_student)
				if self.config.cross_view:
					print(f'Step {global_step}: teacher loss = {teacher_loss:.4f}, student loss = {student_loss:.4f}, time = {t1 - t0:.4f}')
				else:
					print(f'Step {global_step}: train loss = {teacher_loss:.4f}, time = {t1 - t0:.4f}')
				t0 = time.time()
				total_teacher_loss = total_student_loss = 0
				count_teacher = count_student = 0

			# eval dev
			if global_step % self.config.eval_dev_every == 0 or global_step == self.config.max_step:
				print('-' * 20)
				val_loss, uas, las = self.check_dev(self.model, 'teacher')
				history['val_loss'].append(val_loss)
				history['uas'].append(uas)
				history['las'].append(las)
				print(f'EVAL DEV: val loss = {val_loss:.4f}, UAS = {uas:.4f}, LAS = {las:.4f}')
				if las + uas > self.best_las + self.best_uas:
					print('save new best model')
					self.best_las = las
					self.best_uas = uas
					self.best_loss = val_loss
					self.saving_step = global_step
					utils_train.save_model(self, self.config)
					self.best_model = None

				if self.config.print_dev_student and self.config.cross_view:
					for s_i, student_model in enumerate(self.model_students):
						val_loss, uas, las = self.check_dev(student_model, 'student')
						if uas + las > self.best_student + self.best_student * 0.0005:
							print('save new best student')
							utils_train.save_model(self, self.config, 'student')
							self.best_student = uas + las
						print(f'EVAL Student {s_i}: val loss = {val_loss:.4f}, UAS = {uas:.4f}, LAS = {las:.4f}')
				print('-' * 20)
				if global_step - self.saving_step > self.config.max_waiting_step:
					break

			# eval test
			if global_step % self.config.eval_test_every == 0 and global_step != self.config.max_step:
				print('current best step', self.saving_step)
				self.evaluate()
				print('-' * 30)

		torch.save(self.config, self.config.config_file)
		# utils.show_history_graph(history)
		print('finish training')
		print('best uas:', self.best_uas)
		print('best las:', self.best_las)
		print('best step', self.saving_step)
		print('-'*20)
		self.saving_step = self.config.max_step
		utils_train.save_model(self, self.config, 'last')
		self.evaluate(use_best=False)
		self.evaluate()
		if self.config.cross_view:
			for model_index, student_model in enumerate(self.model_students):
				self.evaluate(model_index)
			del self.best_model
			self.best_model = None
			print('-'*20)
			print('best student:')
			self.evaluate(0, use_best=False, use_best_student=True)

	def check_dev(self, model, mode):
		stats = Counter()
		model.eval()
		model.encoder.mode = mode
		dev_batches = self.corpus.dev.batches(self.config.batch_size, shuffle=False, length_ordered=False)
		dev_batch_length = 0
		dev_word_list = []
		dev_length_list = []
		dev_head_list = []
		dev_lab_list = []
		dev_new_order = []
		with torch.no_grad():
			for batch in dev_batches:
				dev_batch_length += 1
				words, index_ids, last_index_position, tags, heads, labels, masks, lengths, origin_words, chars, new_order = batch
				loss, head_list, lab_list = model.predict_batch_with_loss(words, index_ids, last_index_position, tags, chars, heads, labels, masks, lengths)
				stats['val_loss'] += loss.item()
				dev_head_list += head_list
				dev_lab_list += lab_list
				dev_word_list += origin_words
				dev_length_list += lengths
				dev_new_order += new_order

		dev_head_list = utils.unsort(dev_head_list, dev_new_order)
		dev_lab_list = utils.unsort(dev_lab_list, dev_new_order)
		dev_word_list = utils.unsort(dev_word_list, dev_new_order)
		dev_length_list = utils.unsort(dev_length_list, dev_new_order)

		utils.write_conll(self.corpus.vocab, dev_word_list, dev_head_list, dev_lab_list, dev_length_list,
											self.config.parsing_file)
		val_loss = stats['val_loss'] / dev_batch_length
		uas, las = utils.ud_scores(self.config.dev_file, self.config.parsing_file)
		return val_loss, uas, las

	def evaluate(self, model_type=-1, use_best=True, use_best_student=False):  # -1 is teacher
		if use_best:
			if self.best_model is None:
				self.best_model = all_model = torch.load(self.config.model_file, map_location=self.device)
			else:
				all_model = self.best_model
			if model_type == -1:
				model = all_model['model']
				model.encoder = all_model['encoder']
				model.encoder.mode = 'teacher'
			else:
				model = all_model['model_students'][model_type]
				model.encoder = all_model['encoder']
				model.encoder.mode = 'student'
		else:
			if use_best_student:
				all_model = torch.load(self.config.best_student_file, map_location=self.device)
				model = all_model['model_students'][model_type]
				model.encoder = all_model['encoder']
				model.encoder.mode = 'student'
			else:
				model_type = 'last train'
				model = self.model
				model.encoder.mode = 'teacher'
		model.to(self.device)
		print('evaluating', model_type, 'model')
		model.eval()
		test_batches = self.corpus.test.batches(self.config.batch_size, shuffle=False, length_ordered=False)
		test_batch_length = 0
		test_word_list = []
		test_length_list = []
		test_head_list = []
		test_lab_list = []
		gold_head_list = []
		gold_lab_list = []
		pos_list = []
		new_order_list = []
		with torch.no_grad():
			for batch in test_batches:
				test_batch_length += 1
				words, index_ids, last_index_position, tags, heads, labels, masks, lengths, origin_words, chars, new_order = batch
				head_list, lab_list = model.predict_batch(words, index_ids, last_index_position, tags, chars, heads, labels, lengths, masks)
				gold_head_list += [head.data.numpy()[1:lent+1] for head, lent in zip(heads.cpu(), lengths)]
				gold_lab_list += [lab.data.numpy()[1:lent+1] for lab, lent in zip(labels.cpu(), lengths)]
				pos_list += [tag.data.numpy()[1:lent+1] for tag, lent in zip(tags.cpu(), lengths)]
				test_head_list += head_list
				test_lab_list += lab_list
				test_word_list += origin_words  # remove when write conll
				test_length_list += lengths
				new_order_list += new_order

			gold_head_list = utils.unsort(gold_head_list, new_order_list)
			gold_lab_list = utils.unsort(gold_lab_list, new_order_list)
			pos_list = utils.unsort(pos_list, new_order_list)
			test_head_list = utils.unsort(test_head_list, new_order_list)
			test_lab_list = utils.unsort(test_lab_list, new_order_list)
			test_word_list = utils.unsort(test_word_list, new_order_list)
			test_length_list = utils.unsort(test_length_list, new_order_list)

			utils.write_conll(self.corpus.vocab, test_word_list, test_head_list, test_lab_list, test_length_list,
												self.config.parsing_file)
			write_file_error_example(self.config, self.corpus.vocab, test_word_list, test_head_list, gold_head_list, test_lab_list,
															 gold_lab_list, pos_list, test_length_list)
			uas, las = utils.ud_scores(self.config.test_file, self.config.parsing_file)
			print(f'Evaluating Result: UAS = {uas:.4f}, LAS = {las:.4}')

	def annotate(self):
		print('parsing ...')
		input_file = self.config.annotate_file
		unlabel_list = sentence_level.read_unlabel_data(input_file, self.tokenizer, self.corpus.vocab, self.config)
		current_dataset = dataset.Dataset(self.config, unlabel_list, self.corpus.vocab, self.device, self.phobert, origin_ordered=True)
		print(f'parsing {len(current_dataset.lengths)} sentences')

		all_model = torch.load(self.config.model_file, map_location=self.device)
		model = all_model['model']
		model.encoder = all_model['encoder']
		model.encoder.mode = 'teacher'
		model.to(self.device)
		model.eval()

		test_batches = current_dataset.batches(self.config.batch_size, shuffle=False, length_ordered=False)
		test_word_list = []
		test_length_list = []
		test_head_list = []
		test_lab_list = []
		new_order_list = []
		with torch.no_grad():
			for batch in test_batches:
				words, index_ids, last_index_position, tags, heads, labels, masks, lengths, origin_words, chars, new_order = batch
				head_list, lab_list = model.predict_batch(words, index_ids, last_index_position, tags, chars, heads,
														  labels, lengths, masks)
				test_head_list += head_list
				test_lab_list += lab_list
				test_word_list += origin_words  # remove when write conll
				test_length_list += lengths
				new_order_list += new_order

			test_head_list = utils.unsort(test_head_list, new_order_list)
			test_lab_list = utils.unsort(test_lab_list, new_order_list)
			test_word_list = utils.unsort(test_word_list, new_order_list)

			utils.write_conll(self.corpus.vocab, test_word_list, test_head_list, test_lab_list, test_length_list, self.config.annotate_result_file)
			print(f'finish {len(test_length_list)} sentences')
			# for index in range(len(test_length_list)):
			# 	nltk_str = []
			# 	for (w, h, rel) in zip(test_word_list[index], test_head_list[index], test_lab_list[index]):
			# 		relation_type = self.corpus.vocab.i2l[rel]
			# 		if relation_type == 'root':
			# 			relation_type = 'ROOT'
			# 		nltk_str.append(f'{w} _ {h} {relation_type}')
			# 	dot_repr = nltk.DependencyGraph(nltk_str).to_dot()
			# 	source = Source(dot_repr, filename=f'image/dep_tree_{index}', format="png")
			# 	source.view()


def main():
	# load config
	config = Config()
	utils.ensure_dir(config.save_folder)

	# set seed
	torch.manual_seed(config.seed)
	np.random.seed(config.seed)
	random.seed(config.seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(config.seed)

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
