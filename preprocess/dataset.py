import os
import numpy as np
import torch
import time
import math
from collections import defaultdict

from preprocess.sentence_level import preprocess_word, read_data, read_unlabel_data
from preprocess.char import CHAR_DEFAULT, PAD_TOKEN, PAD_INDEX, UNK_TOKEN, UNK_INDEX, ROOT_TOKEN, \
	ROOT_TAG, ROOT_LABEL, ROOT_INDEX, word_format

def wrap(batch, is_float=False, is_bool=False):
	"""Packages the batch as a Variable containing a LongTensor."""
	if is_float:
		wrapping = torch.autograd.Variable(torch.Tensor(batch))
	else:
		wrapping = torch.autograd.Variable(torch.LongTensor(batch))
	if is_bool:
		wrapping = torch.autograd.Variable(torch.BoolTensor(batch))
	if torch.cuda.is_available():
		wrapping = wrapping.cuda()
	return wrapping

def pad(batch, pad_word=PAD_INDEX, is_float=False):
	lens = list(map(len, batch))
	max_len = max(lens)
	padded_batch = []
	for k, seq in zip(lens, batch):
		padded = seq + (max_len - k) * [pad_word]
		padded_batch.append(padded)
	return wrap(padded_batch, is_float)

def pad_char(batch, pad_char=PAD_INDEX, is_float=False):
	# pad word
	max_word_len = max(map(len, [w for sent in batch for w in sent]))
	new_batch = []
	for sent in batch:
		lens = list(map(len, sent))
		new_sent = []
		for k, word in zip(lens, sent):
			padded = word + (max_word_len - k) * [PAD_INDEX]
			new_sent.append(padded)
		new_batch.append(new_sent)
	batch = new_batch

	# pad sentence
	pad_word = max_word_len * [PAD_INDEX]
	lens = list(map(len, batch))
	max_len = max(lens)
	padded_batch = []
	for k, seq in zip(lens, batch):
		padded = seq + (max_len - k) * [pad_word]
		padded_batch.append(padded)
	return wrap(padded_batch, is_float)

def pad_phobert(batch, pad_word=1):
	lens = list(map(len, batch))
	max_len = max(lens)
	padded_batch = []
	for k, seq in zip(lens, batch):
		padded = seq + (max_len - k) * [pad_word]
		padded_batch.append(padded)
	return torch.tensor(padded_batch)

def pad_word_embedding(batch, config):
	pad_word = config.phobert_dim*[PAD_INDEX]
	return pad(batch, pad_word, True)

def pad_mask(batch):
	lens = list(map(len, batch))
	max_len = max(lens)
	padded_batch = []
	for k in lens:
		padded = k * [False] + (max_len - k) * [True]
		padded_batch.append(padded)
	return wrap(padded_batch, False, True)

def default_value():
	return UNK_INDEX

def zero_func():
	return 0

class Vocab:
	def __init__(self, config, sentence_list):
		self.config = config
		self.w2i = defaultdict(default_value)
		self.pre_w = defaultdict(zero_func)
		self.t2i = defaultdict(default_value)
		self.l2i = defaultdict(default_value)
		self.c2i = defaultdict(default_value)

		self.i2w = defaultdict(default_value)
		self.i2t = defaultdict(default_value)
		self.i2l = defaultdict(default_value)
		self.i2c = defaultdict(default_value)

		self.add_word(PAD_TOKEN)
		self.add_word(UNK_TOKEN)
		self.add_word(ROOT_TOKEN)

		self.add_tag(PAD_TOKEN)
		self.add_tag(UNK_TOKEN)
		self.add_tag(ROOT_TAG)

		self.add_label(PAD_TOKEN)
		self.add_label(UNK_TOKEN)
		self.add_label(ROOT_LABEL)

		self.add_char(PAD_TOKEN)
		self.add_char(UNK_TOKEN)
		self.add_char(ROOT_TOKEN)

		for char in CHAR_DEFAULT:
			self.add_char(char)

		for sentence in sentence_list:
			for i in range(sentence.length):
				self.add_word(sentence.word[i])
				if config.pos_type == 'vn':
					self.add_tag(sentence.vn_pos[i])
				else:
					# lab also can be uni
					self.add_tag(sentence.lab_pos[i])
				self.add_label(sentence.dependency_label[i])

	def real_add_word(self, word):
		i = len(self.i2w)
		self.i2w[i] = word
		self.w2i[word] = i

	def add_word(self, word, unk=False, is_label=True):
		word = preprocess_word(word)
		if word not in self.w2i:
			if unk:
				self.i2w[UNK_INDEX] = UNK_TOKEN
				self.w2i[word] = UNK_INDEX
			else:
				if is_label:
					self.real_add_word(word)
				else:
					self.pre_w[word] += 1
					if self.pre_w[word] >= self.config.minimum_frequency:
						self.real_add_word(word)

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

	def add_char(self, char):
		if char not in self.c2i:
			i = len(self.i2c)
			self.i2c[i] = char
			self.c2i[char] = i


class Dataset:
	def __init__(self, config, sentence_list, vocab, device, phobert, origin_ordered=False, cache=False):
		self.phobert = phobert
		self.orgin_ordered = origin_ordered
		self.words = []
		self.tags = []
		self.heads = []
		self.labels = []
		self.lengths = []
		self.origin_words = []
		self.chars = []
		input_ids = []
		last_index_position = []
		self.bucket = []
		self.config = config
		for sentence in sentence_list:
			self.origin_words.append(sentence.word)
			self.lengths.append(sentence.length)
			# self.words.append(sentence.word_embedding)
			tag_list = sentence.lab_pos
			if config.pos_type == 'vn':
				tag_list = sentence.vn_pos
			self.tags.append([vocab.t2i[tag] for tag in tag_list])
			self.heads.append(sentence.head_index)
			self.labels.append([vocab.l2i[label] for label in sentence.dependency_label])
			char_list = []
			for word in sentence.word:
				clear_word = preprocess_word(word, False)
				char_list.append([vocab.c2i[c] for c in clear_word])
			self.chars.append(char_list)
			if config.use_phobert:
				input_ids.append(sentence.input_ids)
				last_index_position.append(sentence.last_index_position)
			self.words.append([vocab.w2i[preprocess_word(word)] for word in sentence.word])

		self.cache = cache
		if config.use_phobert:
			self.input_ids = input_ids
			self.last_index_position = last_index_position
			self.phobert_embs = []
			self.device = device
			if cache:
				self.get_all_embedding()
		else:
			self.input_ids = []
			self.last_index_position = []
		self.init_bucket()

	def init_bucket(self):
		self.order()
		# shuffle into 3 bucket <20, <40 and others
		pivot_20 = pivot_40 = 0
		for index, lent in enumerate(self.lengths):
			if lent <= 20:
				pivot_20 = index
			if lent <= 40:
				pivot_40 = index
		self.bucket = [(0, pivot_20), (pivot_20, pivot_40), (pivot_40, len(self.lengths))]

	def get_all_embedding(self):
		self.phobert_embs = []
		n = len(self.lengths)
		batch_order = list(range(0, n, self.config.batch_size))
		for i in batch_order:
			self.phobert_embs += self.get_phobert_embedding(i, i+self.config.batch_size)

	def get_phobert_embedding(self, begin_position, end_position):
		self.phobert.eval()
		n = len(self.input_ids)
		batch_input_ids = self.input_ids[begin_position:end_position]
		padded_input_ids = pad_phobert(batch_input_ids).to(self.device)
		with torch.no_grad():
			origin_features = self.phobert(padded_input_ids)
			# hidden layer format: [layer: 13(+1 output)][batch][ids][768]
			features = origin_features[2][self.config.phobert_layer]
			# attention format: [layer: 12][batch][head: 12][ids][ids]
			# attention_heads = utils.get_attention_heads(origin_features[3], self.config.attention_requires, self.config.attention_head_tops)
		words = []
		for sentence_index in range(begin_position, min(n, end_position)):
			# get embedding of each word
			word_embedding = []
			last_index_position_list = self.last_index_position[sentence_index]
			for word_index in range(len(last_index_position_list) - 2):
				start_index = last_index_position_list[word_index]
				end_index = last_index_position_list[word_index+1]
				if self.config.phobert_subword == 'first':
					end_index = start_index + 1
				word_emb = features[sentence_index-begin_position][start_index:end_index]
				word_emb = torch.sum(word_emb, 0).cpu().data.numpy().tolist()
				# word_emb.append(word_format(self.origin_words[sentence_index][word_index], word_index))
				word_embedding.append(word_emb)
			words.append(word_embedding)
		return words

	def swap_data(self, new_order):
		if len(self.words) == len(new_order):
			self.words = [self.words[i] for i in new_order]
		if self.config.use_phobert:
			self.input_ids = [self.input_ids[i] for i in new_order]
			self.last_index_position = [self.last_index_position[i] for i in new_order]
			if self.cache:
				self.phobert_embs = [self.phobert_embs[i] for i in new_order]
		self.chars = [self.chars[i] for i in new_order]
		self.tags = [self.tags[i] for i in new_order]
		self.heads = [self.heads[i] for i in new_order]
		self.labels = [self.labels[i] for i in new_order]
		self.lengths = [self.lengths[i] for i in new_order]
		self.origin_words = [self.origin_words[i] for i in new_order]

	def order(self):
		if self.orgin_ordered or len(self.lengths) < 2:
			return
		old_order = zip(range(len(self.lengths)), self.lengths)
		new_order, _ = zip(*sorted(old_order, key=lambda t: t[1]))
		self.swap_data(new_order)

	def order_batch(self, batch_size):
		origin_ids = []
		n = len(self.lengths)
		batch_list = list(range(0, n, batch_size))
		for i in batch_list:
			old_order = list(zip(range(i, min(i+batch_size, n)), self.lengths[i:i+batch_size]))
			old_order.sort(reverse=True, key=lambda t: t[1])
			new_order = [item0 for item0, item1 in old_order]
			origin_ids += new_order
		self.swap_data(origin_ids)
		old_order = list(zip(range(n), origin_ids))
		old_order.sort(key=lambda t: t[1])
		new_order = [item0 for item0, item1 in old_order]
		return new_order

	def shuffle(self):
		if self.orgin_ordered:
			return
		self.order()
		new_order = []
		for start_index, end_index in self.bucket:
			temp_order = list(range(start_index, end_index))
			np.random.shuffle(temp_order)
			new_order.append(temp_order)
		new_order = [position for order_list in new_order for position in order_list]
		self.swap_data(new_order)

	def batches(self, batch_size, shuffle=True, length_ordered=False):
		"""An iterator over batches."""
		n = len(self.lengths)
		batch_order = list(range(0, n, batch_size))
		if shuffle:
			self.shuffle()
			np.random.shuffle(batch_order)
		if length_ordered:
			self.order()
		#index_ids = last_index_position = phobert_emb = []
		new_order = self.order_batch(batch_size)
		for i in batch_order:
			words = pad(self.words[i:i + batch_size])
			if self.config.use_phobert:
				#index_ids = pad_phobert(self.input_ids[i:i + batch_size]).to(self.device)
				last_index_position = pad(self.last_index_position[i:i + batch_size])
				if self.cache:
					phobert_emb = pad_word_embedding(self.phobert_embs[i:i+batch_size], self.config)
				else:
					phobert_emb = pad_word_embedding(self.get_phobert_embedding(i, i+batch_size), self.config)
			chars = pad_char(self.chars[i:i + batch_size])
			tags = pad(self.tags[i:i + batch_size])
			heads = pad(self.heads[i:i + batch_size])
			labels = pad(self.labels[i:i + batch_size])
			masks = pad_mask(self.labels[i:i + batch_size])
			lengths = self.lengths[i:i + batch_size]
			origin_words = self.origin_words[i:i + batch_size]
			yield words, phobert_emb, last_index_position, tags, heads, labels, masks, lengths, origin_words, chars, new_order[i: i+batch_size]
		# return origin order
		self.swap_data(new_order)

	def concat(self, other):
		self.words += other.words
		self.tags += other.tags
		self.heads += other.heads
		self.labels += other.labels
		self.lengths += other.lengths
		self.origin_words += other.origin_words
		self.chars += other.chars
		self.input_ids += other.input_ids
		self.last_index_position += other.last_index_position
		self.phobert_embs += other.phobert_embs


class Corpus:
	def __init__(self, config, device, tokenizer, phobert):
		train_list = read_data(config.pos_train_file, tokenizer, config)
		dev_list = read_data(config.pos_dev_file, tokenizer, config)
		test_list = read_data(config.pos_test_file, tokenizer, config)

		if config.train_percent < 1:
			real_train = int(len(train_list) * config.train_percent)
			train_list = train_list[:real_train]
			print('sentence use for train', real_train)

		if os.path.exists(config.vocab_file) and config.continue_train:
			self.vocab = torch.load(config.vocab_file)
			config.add_more_vocab = False
		else:
			self.vocab = Vocab(config, train_list)
			torch.save(self.vocab, config.vocab_file)
		if config.mode == 'train':
			self.train = Dataset(config, train_list, self.vocab, device, phobert, False, cache=True)
			self.dev = Dataset(config, dev_list, self.vocab, device, phobert, True)
		self.test = Dataset(config, test_list, self.vocab, device, phobert, True)

class Unlabel_Corpus:
	def __init__(self, config, device, vocab, tokenizer, phobert):
		self.config = config
		self.dataset = Dataset(config, [], vocab, device, phobert, False)
		for file_name in os.listdir(config.unlabel_folder):
			print('preprocess file:', file_name)
			if file_name.endswith('.txt') is False:
				continue
			input_file = os.path.join(config.unlabel_folder, file_name)
			unlabel_list = read_unlabel_data(input_file, tokenizer, vocab, config)
			current_dataset = Dataset(config, unlabel_list, vocab, device, phobert, False)
			self.dataset.concat(current_dataset)
		self.dataset.init_bucket()
		print('total length unlabel corpus:', len(self.dataset.lengths))





