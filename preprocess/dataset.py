import os
import numpy as np
import torch
from collections import defaultdict

from preprocess.sentence_level import preprocess_word, read_data, read_unlabel_data
from preprocess.char import CHAR_DEFAULT, PAD_TOKEN, PAD_INDEX, UNK_TOKEN, UNK_INDEX, ROOT_TOKEN, ROOT_TAG, ROOT_LABEL, ROOT_INDEX

def wrap(batch, is_float=False):
	"""Packages the batch as a Variable containing a LongTensor."""
	if is_float:
		wrapping = torch.autograd.Variable(torch.Tensor(batch))
	else:
		wrapping = torch.autograd.Variable(torch.LongTensor(batch))
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
		padded = k * [1] + (max_len - k) * [0]
		padded_batch.append(padded)
	return wrap(padded_batch, True)

def default_value():
	return UNK_INDEX

class Vocab:
	def __init__(self, config, sentence_list):
		self.w2i = defaultdict(default_value)
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
				if config.use_vn_pos:
					self.add_tag(sentence.vn_pos[i])
				else:
					self.add_tag(sentence.ud_pos[i])
				self.add_label(sentence.dependency_label[i])

	def add_word(self, word, unk=False):
		word = preprocess_word(word)
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

	def add_char(self, char):
		if char not in self.c2i:
			i = len(self.i2c)
			self.i2c[i] = char
			self.c2i[char] = i


class Dataset:
	def __init__(self, config, sentence_list, vocab, phobert, device, origin_ordered=False, cache_embedding=False):
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
		self.cache_embedding = cache_embedding
		for sentence in sentence_list:
			self.origin_words.append(sentence.word)
			self.lengths.append(sentence.length)
			# self.words.append(sentence.word_embedding)
			tag_list = sentence.vn_pos
			if not config.use_vn_pos:
				tag_list = sentence.ud_pos
			self.tags.append([vocab.t2i[tag] for tag in tag_list])
			self.heads.append(sentence.head_index)
			self.labels.append([vocab.l2i[label] for label in sentence.dependency_label])
			char_list = []
			for word in sentence.word:
				clear_word = preprocess_word(word)
				char_list.append([vocab.c2i[c] for c in clear_word])
			self.chars.append(char_list)
			if config.use_phobert:
				input_ids.append(sentence.input_ids)
				last_index_position.append(sentence.last_index_position)
			else:
				self.words.append([vocab.w2i[preprocess_word(word)] for word in sentence.word])

		# self.process_embedding(phobert, input_ids, last_index_position, device)
		if config.use_phobert:
			self.phobert = phobert
			self.input_ids = input_ids
			self.last_index_position = last_index_position
			self.device = device
			if cache_embedding:
				self.get_all_embedding()
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
		self.words = []
		n = len(self.lengths)
		batch_order = list(range(0, n, self.config.batch_size))
		for i in batch_order:
			self.words += self.get_phobert_embedding(i, i+self.config.batch_size)

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
				word_emb = features[sentence_index-begin_position][start_index:end_index]
				# word_embedding.append(torch.sum(word_emb, 0).numpy() / (end_index-start_index))
				word_embedding.append(torch.sum(word_emb, 0).cpu().data.numpy().tolist())
			words.append(word_embedding)
		return words

	def swap_data(self, new_order):
		if len(self.words) == len(new_order):
			self.words = [self.words[i] for i in new_order]
		if self.config.use_phobert:
			self.input_ids = [self.input_ids[i] for i in new_order]
			self.last_index_position = [self.last_index_position[i] for i in new_order]
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
		for i in batch_order:
			if self.config.use_phobert:
				if self.cache_embedding:
					words = pad_word_embedding(self.words[i:i + batch_size], self.config)
				else:
					words = pad_word_embedding(self.get_phobert_embedding(i, i+batch_size), self.config)
			else:
				words = pad(self.words[i:i + batch_size])
			chars = pad_char(self.chars[i:i + batch_size])
			tags = pad(self.tags[i:i + batch_size])
			heads = pad(self.heads[i:i + batch_size])
			labels = pad(self.labels[i:i + batch_size])
			masks = pad_mask(self.labels[i:i + batch_size])
			lengths = self.lengths[i:i + batch_size]
			origin_words = self.origin_words[i:i + batch_size]
			yield words, tags, heads, labels, masks, lengths, origin_words, chars

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


class Corpus:
	def __init__(self, config, device, phobert, tokenizer):
		train_list = read_data(config.train_file, tokenizer)
		dev_list = read_data(config.dev_file, tokenizer)
		test_list = read_data(config.test_file, tokenizer)

		if os.path.exists(config.vocab_file):
			self.vocab = torch.load(config.vocab_file)
		else:
			self.vocab = Vocab(config, train_list + dev_list + test_list)
		self.train = Dataset(config, train_list, self.vocab, phobert, device, False, cache_embedding=True)
		self.dev = Dataset(config, dev_list, self.vocab, phobert, device, True, cache_embedding=True)
		self.test = Dataset(config, test_list, self.vocab, phobert, device, True, cache_embedding=True)

class Unlabel_Corpus:
	def __init__(self, config, device, vocab, phobert, tokenizer):
		self.config = config
		self.dataset = Dataset(config, [], vocab, phobert, device)
		for file_name in os.listdir(config.unlabel_folder):
			print('preprocess file:', file_name)
			if file_name.endswith('.txt') is False:
				continue
			input_file = os.path.join(config.unlabel_folder, file_name)
			unlabel_list = read_unlabel_data(input_file, tokenizer, vocab)
			current_dataset = Dataset(config, unlabel_list, vocab, phobert, device, False)
			self.dataset.concat(current_dataset)
		self.dataset.init_bucket()
		print('total length unlabel corpus:', len(self.dataset.lengths))





