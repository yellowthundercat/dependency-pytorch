import torchtext
import torch
from utils import utils

from torch.utils.data import Dataset, DataLoader

def read_data(filename, config, data_fields):
	input_file = open(filename, encoding='utf-8')
	sentence_list = []
	words = []
	postags = []
	heads = []
	de_label = []
	for line in input_file:
		if line.startswith('#'):  # skip comment line
			continue
		line = line.strip()
		if line == '' or line == '\n':
			if len(words) > 1:
				sentence_list.append(torchtext.data.Example.fromlist([words, postags, heads], data_fields))
				words = []
				postags = []
				heads = []
				de_label = []
		else:
			columns = line.split('\t')
			words.append(columns[1])
			if config.use_vn_pos:
				postags.append(columns[4])
			else:
				postags.append(columns[3])
			heads.append(int(columns[6]))
			de_label.append(columns[7])
	if len(words) > 1:
		sentence_list.append(torchtext.data.Example.fromlist([words, postags, heads], data_fields))
	utils.log('Read file:', filename)
	utils.log('Number of sentence:', len(sentence_list))
	return torchtext.data.Dataset(sentence_list, data_fields)

class Sentence:
	def __init__(self, word_list):
		self.word = word_list[0]
		self.ud_pos = word_list[1]
		self.vn_pos = word_list[2]
		self.head_index = word_list[3]
		self.dependency_label = word_list[4]

	def __str__(self):
		return ' '.join(self.word)

def _get_useful_column_ud(sentence, data_fields):
	# word, ud pos, vn pos, head index, dependency label
	# return [Sentence([word[1], word[3], word[4], word[6], word[7]]) for word in sentence]
	return [torchtext.data.Example.fromlist([word[1], word[3], word[4], word[6], word[7]], data_fields) for word in sentence]



