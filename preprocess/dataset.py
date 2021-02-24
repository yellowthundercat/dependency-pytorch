from torch.utils.data import Dataset, DataLoader
import torchtext
from utils import utils


def get_data_loader(filename, config):
	dataset = DependencyDataset(filename)
	data_loader = DataLoader(dataset,  shuffle=True, num_workers=2)
	return data_loader

class Sentence:
	def __init__(self, word_list):
		self.word = word_list[0]
		self.ud_pos = word_list[1]
		self.vn_pos = word_list[2]
		self.head_index = word_list[3]
		self.dependency_label = word_list[4]

	def __str__(self):
		return ' '.join(self.word)

def _get_useful_column_ud(sentence):
	# word, ud pos, vn pos, head index, dependency label
	# return [Sentence([word[1], word[3], word[4], word[6], word[7]]) for word in sentence]
	return [ for word in sentence]

class DependencyDataset(Dataset):
	def __init__(self, filename):
		input_file = open(filename, encoding='utf-8')
		sentence_list = []
		sentence = []
		for line in input_file:
			if line.startswith('#'):  # skip comment line
				continue
			if line == '' or line == '\n':
				if len(sentence) > 1:
					sentence_list.append(_get_useful_column_ud(sentence))
					sentence = []
			else:
				sentence.append(line.split('\t'))
		if len(sentence) > 1:
			sentence_list.append(_get_useful_column_ud(sentence))
		self.sentence_list = sentence_list
		utils.log('Read file:', filename)
		utils.log('Number of sentence:', len(sentence_list))

	def __len__(self):
		return len(self.sentence_list)

	def __getitem__(self, idx):
		return self.sentence_list[idx]


