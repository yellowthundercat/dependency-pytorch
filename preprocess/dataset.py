import os
import re
import numpy as np
import torch
from utils import utils
from collections import defaultdict
from transformers import AutoModel, AutoTokenizer

PAD_TOKEN = '<pad>'
PAD_INDEX = 0

UNK_TOKEN = '<unk>'
UNK_INDEX = 1

ROOT_TOKEN = '<root>'
ROOT_TAG = 'ROOT'
ROOT_LABEL = '_root_'
ROOT_INDEX = 2

def wrap(batch):
	"""Packages the batch as a Variable containing a LongTensor."""
	wrapping = torch.autograd.Variable(torch.LongTensor(batch))
	if torch.cuda.is_available():
		wrapping = wrapping.cuda()
	return wrapping

def pad(batch, pad_word=PAD_INDEX):
	lens = list(map(len, batch))
	max_len = max(lens)
	padded_batch = []
	for k, seq in zip(lens, batch):
		padded = seq + (max_len - k) * [pad_word]
		padded_batch.append(padded)
	return wrap(padded_batch)

def pad_word_embedding(batch, config):
	pad_word = config.word_emb_dim*[PAD_INDEX]
	return pad(batch, pad_word)

def pad_mask(batch):
	lens = list(map(len, batch))
	max_len = max(lens)
	padded_batch = []
	for k in lens:
		padded = k * [1] + (max_len - k) * [0]
		padded_batch.append(padded)
	return wrap(padded_batch)

def read_data(filename, tokenizer, phobert):
	sentence_count = 0
	input_file = open(filename, encoding='utf-8')
	sentence_list = []
	sentence = []
	for line in input_file:
		if line.startswith('#'):  # skip comment line
			continue
		line = line.strip()
		if line == '' or line == '\n':
			if len(sentence) > 1:
				sentence_list.append(_get_useful_column_ud(sentence, tokenizer, phobert))
				sentence = []
				sentence_count += 1
				if sentence_count % 100 == 0:
					print(f'loading finish: {sentence_count} sentence')
		else:
			sentence.append(line.split('\t'))
	if len(sentence) > 1:
		sentence_list.append(_get_useful_column_ud(sentence, tokenizer, phobert))
	utils.log('Read file:', filename)
	utils.log('Number of sentence:', len(sentence_list))
	return sentence_list


def preprocess_word(word):
	return re.sub(r'\d', '0', word.lower())

class Sentence:
	def __init__(self, word_list, ud_pos_list, vn_pos_list, head_list, dependency_list, tokenizer, phobert):
		cls_id = 0
		sep_id = 2
		input_ids = [cls_id]
		last_index_position_list = [1]
		# get encode index from word
		for word in word_list:
			token = tokenizer.encode(preprocess_word(word))
			input_ids += token[1:(len(token)-1)]
			last_index_position_list.append(len(input_ids))
		input_ids.append(sep_id)
		# get embedding of full sentence
		input_ids = torch.tensor([input_ids])
		with torch.no_grad():
			features = phobert(input_ids)[0][0]
		# get embedding of each word
		self.word_embedding = []
		for index, word in enumerate(word_list):
			word_emb = features[last_index_position_list[index]:last_index_position_list[index+1]]
			self.word_embedding.append(torch.sum(word_emb, 0))

		self.word = word_list
		self.ud_pos = ud_pos_list
		self.vn_pos = vn_pos_list
		self.head_index = head_list
		self.dependency_label = dependency_list
		self.length = len(self.head_index)

	def __str__(self):
		return ' '.join(self.word)

def _get_useful_column_ud(sentence, tokenizer, phobert):
	# word, ud pos, vn pos, head index, dependency label
	sentence = [[0, ROOT_TOKEN, 2, ROOT_TAG, ROOT_TAG, 5, 0, ROOT_LABEL, 8]] + sentence
	word_list = []
	ud_pos_list = []
	vn_pos_list = []
	head_index_list = []
	dependency_label_list = []
	for word in sentence:
		word_list.append(word[1])
		ud_pos_list.append(word[3])
		vn_pos_list.append(word[4])
		head_index_list.append(int(word[6]))
		dependency_label_list.append(word[7])
	return Sentence(word_list, ud_pos_list, vn_pos_list, head_index_list, dependency_label_list, tokenizer, phobert)

class Vocab:
	def __init__(self, config, sentence_list):
		self.w2i = defaultdict(lambda: UNK_INDEX)
		self.t2i = defaultdict(lambda: UNK_INDEX)
		self.l2i = defaultdict(lambda: UNK_INDEX)

		self.i2w = defaultdict(lambda: UNK_TOKEN)
		self.i2t = defaultdict(lambda: UNK_TOKEN)
		self.i2l = defaultdict(lambda: UNK_TOKEN)

		self.add_word(PAD_TOKEN)
		self.add_word(UNK_TOKEN)
		self.add_word(ROOT_TOKEN)

		self.add_tag(PAD_TOKEN)
		self.add_tag(UNK_TOKEN)
		self.add_tag(ROOT_TAG)

		self.add_label(PAD_TOKEN)
		self.add_label(UNK_TOKEN)
		self.add_label(ROOT_LABEL)

		for sentence in sentence_list:
			for i in range(sentence.length):
				self.add_word(sentence.word[i])
				if config.use_vn_pos:
					self.add_tag(sentence.vn_pos[i])
				else:
					self.add_tag(sentence.ud_pos[i])
				self.add_label(sentence.dependency_label[i])

	def add_word(self, word, unk=False):
		if word not in self.w2i:
			if unk:
				self.i2w[UNK_INDEX] = UNK_TOKEN
				self.w2i[word] = UNK_INDEX
			else:
				i = len(self.i2w)
				self.i2w[i] = word
				self.w2i[word] = i

	def add_tag(self, tag):
		if tag not in self.t2i:
			i = len(self.i2t)
			self.i2t[i] = tag
			self.t2i[tag] = i

	def add_label(self, label):
		if label not in self.l2i:
			i = len(self.i2l)
			self.i2l[i] = label
			self.l2i[label] = i


class Dataset:
	def __init__(self, config, sentence_list, vocab):
		self.words = []
		self.tags = []
		self.heads = []
		self.labels = []
		self.lengths = []
		self.config = config
		for sentence in sentence_list:
			self.lengths.append(sentence.length)
			self.words.append(sentence.word_embedding)
			tag_list = sentence.vn_pos
			if not config.use_vn_pos:
				tag_list = sentence.ud_pos
			self.tags.append([vocab.t2i(tag) for tag in tag_list])
			self.heads.append(sentence.head_index)
			self.labels.append([vocab.l2i(label) for label in sentence.dependency_label])

	def order(self):
		old_order = zip(range(len(self.lengths)), self.lengths)
		new_order, _ = zip(*sorted(old_order, key=lambda t: t[1]))
		self.words = [self.words[i] for i in new_order]
		self.tags = [self.tags[i] for i in new_order]
		self.heads = [self.heads[i] for i in new_order]
		self.labels = [self.labels[i] for i in new_order]
		self.lengths = [self.lengths[i] for i in new_order]

	def shuffle(self):
		n = len(self.words)
		new_order = list(range(0, n))
		np.random.shuffle(new_order)
		self.words = [self.words[i] for i in new_order]
		self.tags = [self.tags[i] for i in new_order]
		self.heads = [self.heads[i] for i in new_order]
		self.labels = [self.labels[i] for i in new_order]
		self.lengths = [self.lengths[i] for i in new_order]

	def batches(self, batch_size, shuffle=True, length_ordered=False):
		"""An iterator over batches."""
		n = len(self.words)
		batch_order = list(range(0, n, batch_size))
		if shuffle:
			self.shuffle()
			np.random.shuffle(batch_order)
		if length_ordered:
			self.order()
		for i in batch_order:
			words = pad_word_embedding(self.words[i:i + batch_size], self.config)
			tags = pad(self.tags[i:i + batch_size])
			heads = pad(self.heads[i:i + batch_size])
			labels = pad(self.labels[i:i + batch_size])
			masks = pad_mask(self.labels[i:i + batch_size])
			yield words, tags, heads, labels, masks

class Corpus:
	def __init__(self, config):
		print('preparing corpus')
		phobert = AutoModel.from_pretrained("vinai/phobert-base")
		tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

		train_list = read_data(config.train_file, tokenizer, phobert)
		dev_list = read_data(config.dev_file, tokenizer, phobert)
		test_list = read_data(config.test_file, tokenizer, phobert)

		if os.path.exists(config.vocab_file):
			self.vocab = torch.load(config.vocab_file)
		else:
			self.vocab = Vocab(config, train_list + dev_list + test_list)
		self.train = Dataset(config, train_list, self.vocab)
		self.dev = Dataset(config, dev_list, self.vocab)
		# self.test = Dataset(config, test_list, self.vocab)




