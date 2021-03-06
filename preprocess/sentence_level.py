import re
import time
from preprocess.char import ROOT_TOKEN, ROOT_TAG, ROOT_LABEL
from utils import utils

def _get_useful_column_ud(sentence, tokenizer, config):
	# word, ud pos, vn pos, head index, dependency label
	sentence = [[0, ROOT_TOKEN, 2, ROOT_TAG, ROOT_TAG, 5, 0, ROOT_LABEL, 8, 9, ROOT_TAG]] + sentence
	word_list = []
	vn_pos_list = []
	lab_pos_list = []
	head_index_list = []
	dependency_label_list = []
	for word in sentence:
		word_list.append(word[1])
		vn_pos_list.append(word[4])
		head_index_list.append(int(word[6]))
		dependency_label_list.append(word[7])
		if config.pos_type == 'vn':
			lab_pos_list.append('-')
		else:
			lab_pos_list.append(word[10])
	return Sentence(word_list, vn_pos_list, lab_pos_list, head_index_list, dependency_label_list, tokenizer)

def read_data(filename, tokenizer, config):
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
				sentence_list.append(_get_useful_column_ud(sentence, tokenizer, config))
				sentence = []
				sentence_count += 1
		else:
			sentence.append(line.split('\t'))
	if len(sentence) > 1:
		sentence_list.append(_get_useful_column_ud(sentence, tokenizer, config))
	utils.log('Read file:', filename)
	utils.log('Number of sentence:', len(sentence_list))
	return sentence_list

def unlabel_sentence(word_list, pos_list, tokenizer):
	word_list = [ROOT_TOKEN] + word_list
	pos_list = [ROOT_TAG] + pos_list
	lent = len(word_list)
	return Sentence(word_list, ['0']*lent, pos_list, [0]*lent, [0]*lent, tokenizer)

def read_unlabel_data(file_name, tokenizer, vocab):
	sentence_list = []
	input_file = open(file_name, encoding='utf-8')
	for sentence in input_file:
		word_list = []
		pos_list = []
		token_list = sentence[:-1].split(' ')
		for token in token_list:
			pos = token.split('/')[-1]
			word_part = token[:len(token) - len(pos) - 1]
			vocab.add_word(word_part, unk=False, is_label=False)
			word_list.append(word_part)
			pos_list.append(pos)
		if 2 < len(word_list) < 50:
			sentence_list.append(unlabel_sentence(word_list, pos_list, tokenizer))
	return sentence_list

def preprocess_word(word, is_lower=True):
	if is_lower:
		return re.sub(r'\d', '0', word.lower())
	return re.sub(r'\d', '0', word)

class Sentence:
	def __init__(self, word_list, vn_pos_list, lab_pos_list, head_list, dependency_list, tokenizer):
		if tokenizer:
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
			last_index_position_list.append(len(input_ids))
			# get embedding of full sentence
			# input_ids = torch.tensor([input_ids])
			self.input_ids = input_ids
			self.last_index_position = last_index_position_list

		self.word = word_list
		self.vn_pos = [pos.lower() for pos in vn_pos_list]
		self.lab_pos = [pos.lower() for pos in lab_pos_list]
		self.head_index = head_list
		self.dependency_label = dependency_list
		self.length = len(self.head_index)

	def __str__(self):
		return ' '.join(self.word)

