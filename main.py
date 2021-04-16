from collections import defaultdict, Counter
import os
import time
import torch
from transformers import AutoModel, AutoTokenizer

from config.default_config import Config
from utils import utils
from preprocess import dataset
from utils.find_error_example import write_file_error_example
from utils import utils_train

class DependencyParser:
	def __init__(self, config):
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.config = config

		# word embedding
		print('preprocess corpus')
		phobert = tokenizer = None
		if config.use_phobert:
			phobert = AutoModel.from_pretrained("vinai/phobert-base", output_hidden_states=True).to(self.device)
			tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
		self.corpus = dataset.Corpus(config, self.device, phobert, tokenizer)
		if config.cross_view and config.mode == 'train':
			print('prepare unlabel data')
			self.unlabel_corpus = dataset.Unlabel_Corpus(config, self.device, self.corpus.vocab, phobert, tokenizer)
		print('total vocab word', len(self.corpus.vocab.w2i))
		print('total vocab character', len(self.corpus.vocab.c2i))

		# model
		if os.path.exists(config.model_file) and config.continue_train:
			print('We will continue training')
			all_model = torch.load(config.model_file)
			utils_train.load_model(self, all_model, config)
		else:
			print('We will train model from scratch')
			utils_train.init_model(self, config)
		self.model.to(self.device)
		if config.cross_view:
			for model_student in self.model_students:
				model_student.to(self.device)

	def train_student(self, unlabel_batch):
		# use teacher to predict
		self.model.eval()
		words, phobert_emds, tags, heads, labels, masks, lengths, origin_words, chars = unlabel_batch
		head_list, lab_list = self.model.predict_batch(words, phobert_emds, tags, chars, lengths)
		heads = dataset.pad([head.tolist() for head in head_list])
		labels = dataset.pad([lab.tolist() for lab in lab_list])

		# train student
		total_loss = 0
		for student_model, student_optimizer, student_scheduler in zip(self.model_students, self.optimizer_students, self.scheduler_students):
			student_model.train()
			student_model.encoder.mode = 'student'
			loss = self.model(words, phobert_emds, tags, chars, heads, labels, masks)
			student_optimizer.zero_grad()
			loss.backward()
			student_optimizer.step()
			student_scheduler.step()
			total_loss += loss.item()
		return total_loss

	def train_teacher(self, train_batch):
		self.model.train()
		self.model.encoder.mode = 'teacher'
		words, phobert_embs, tags, heads, labels, masks, lengths, origin_words, chars = train_batch
		loss = self.model(words, phobert_embs, tags, chars, heads, labels, masks)
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()
		if self.config.use_momentum:
			self.scheduler.step()
		return loss.item()

	def train(self):
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
				#train teacher]
				try:
					train_batch = next(train_batches)
				except StopIteration:
					train_batches = self.corpus.train.batches(self.config.batch_size, length_ordered=self.config.length_ordered)
					train_batch = next(train_batches)
				total_teacher_loss += self.train_teacher(train_batch)
				count_teacher += 1
			else:
				# train student
				try:
					unlabel_batch = next(unlabel_batches)
				except StopIteration:
					unlabel_batches = self.unlabel_corpus.dataset.batches(self.config.batch_size, length_ordered=self.config.length_ordered)
					unlabel_batch = next(unlabel_batches)
				total_student_loss += self.train_student(unlabel_batch)
				count_student += 5

			# print result
			if global_step % self.config.print_step == 0 or global_step == self.config.max_step:
				t1 = time.time()
				teacher_loss = total_teacher_loss/max(1, count_teacher)
				student_loss = total_student_loss/max(1, count_student)
				if self.config.cross_view:
					print(f'Step {global_step}: teacher loss = {teacher_loss:.4f}, student loss = {student_loss:.4f}, time = {t1 - t0:.4f}')
				else:
					print(
						f'Step {global_step}: train loss = {teacher_loss:.4f}, time = {t1 - t0:.4f}')
				t0 = time.time()
				total_teacher_loss = total_student_loss = 0
				count_teacher = count_student = 0

			# eval dev
			if global_step % self.config.eval_dev_every == 0 or global_step == self.config.max_step:
				print('-' * 20)
				val_loss, uas, las = self.check_dev()
				history['val_loss'].append(val_loss)
				history['uas'].append(uas)
				history['las'].append(las)
				print(f'EVAL DEV: val loss = {val_loss:.4f}, UAS = {uas:.4f}, LAS = {las:.4f}')
				if uas + las > self.best_uas + self.best_las:
					print('save new best model')
					self.best_las = las
					self.best_uas = uas
					self.best_loss = val_loss
					self.saving_step = global_step
					utils_train.save_model(self, self.config)
				print('-' * 20)
				if global_step - self.saving_step > self.config.max_waiting_step:
					break

		# torch.save(self.config, self.config.config_file)
		# utils.show_history_graph(history)
		print('finish training')
		print('best uas:', self.best_uas)
		print('best las:', self.best_las)
		print('best step', self.saving_step)
		print('-'*20)
		self.evaluate()
		if self.config.cross_view:
			for model_index in range(5):
				self.evaluate(model_index)

	def check_dev(self):
		stats = Counter()
		self.model.eval()
		dev_batches = self.corpus.dev.batches(self.config.batch_size, shuffle=False, length_ordered=False)
		dev_batch_length = 0
		dev_word_list = []
		dev_length_list = []
		dev_head_list = []
		dev_lab_list = []
		with torch.no_grad():
			for batch in dev_batches:
				dev_batch_length += 1
				words, phobert_embs, tags, heads, labels, masks, lengths, origin_words, chars = batch
				loss, head_list, lab_list = self.model.predict_batch_with_loss(words, phobert_embs, tags, chars, heads, labels, masks, lengths)
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

	def evaluate(self, model_type=-1):  # -1 is teacher
		all_model = torch.load(self.config.model_file)
		if model_type == -1:
			self.model = all_model['model']
		else:
			self.model = all_model['model_students'][model_type]
		self.model.to(self.device)
		print('evaluating', model_type, 'model')
		self.model.eval()
		test_batches = self.corpus.test.batches(self.config.batch_size, shuffle=False, length_ordered=False)
		test_batch_length = 0
		test_word_list = []
		test_length_list = []
		test_head_list = []
		test_lab_list = []
		gold_head_list = []
		gold_lab_list = []
		pos_list = []
		with torch.no_grad():
			for batch in test_batches:
				test_batch_length += 1
				words, phobert_embs, tags, heads, labels, masks, lengths, origin_words, chars = batch
				head_list, lab_list = self.model.predict_batch(words, phobert_embs, tags, chars, lengths)
				gold_head_list += [head.data.numpy()[:lent] for head, lent in zip(heads.cpu(), lengths)]
				gold_lab_list += [lab.data.numpy()[:lent] for lab, lent in zip(labels.cpu(), lengths)]
				pos_list += [tag.data.numpy()[:lent] for tag, lent in zip(tags.cpu(), lengths)]
				test_head_list += head_list
				test_lab_list += lab_list
				test_word_list += origin_words
				test_length_list += lengths
			utils.write_conll(self.corpus.vocab, test_word_list, test_head_list, test_lab_list, test_length_list,
												self.config.parsing_file)
			write_file_error_example(self.config, self.corpus.vocab, test_word_list, test_head_list, gold_head_list, test_lab_list,
															 gold_lab_list, pos_list, test_length_list)
			uas, las = utils.ud_scores(self.config.test_file, self.config.parsing_file)
			print(f'Evaluating Result: UAS = {uas:.4f}, LAS = {las:.4}')

	def annotate(self):
		print('parsing')
		input_file = open(self.config.annotate_file, encoding='utf-8')
		sentence = input_file.read()
		# waiting pos



def main():
	# load config
	config = Config()
	utils.ensure_dir(config.save_folder)
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
