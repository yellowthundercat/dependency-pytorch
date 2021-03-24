import os
import sys
import time
import matplotlib.pyplot as plt
from utils import conll18_ud_eval

def ensure_dir(d, alert=True):
	if not os.path.exists(d):
		if alert:
			print("Directory {} do not exist; creating".format(d))
		os.makedirs(d)

# print log
def log(*args):
	msg = ' '.join(map(str, args))
	sys.stdout.write(msg + '\n')
	sys.stdout.flush()

def heading(*args):
	log()
	log(80 * '=')
	log(*args)
	log(80 * '=')

def show_history_graph(history):
	plt.plot(history['train_loss'])
	plt.plot(history['val_loss'])
	plt.plot(history['uas'])
	plt.legend(['training loss', 'validation loss', 'UAS'])
	plt.show()

def write_conll(vocab, words, head_list, lab_list, lengths, file_name):
	output_file = open(file_name, 'w', encoding='utf-8')
	for index, sentence_length in enumerate(lengths):
		word_index = 0
		for word, head, lab in zip(words[index], head_list[index], lab_list[index]):
			if word_index > 0:
				output_file.write(f'{word_index}\t{word}\t{word}\t-\t-\t-\t{head}\t{vocab.i2l[lab]}\t-\t-\n')
			word_index += 1
		output_file.write('\n')

# ud utils
def ud_scores(gold_conllu_file, system_conllu_file):
	gold_ud = conll18_ud_eval.load_conllu_file(gold_conllu_file)
	system_ud = conll18_ud_eval.load_conllu_file(system_conllu_file)
	evaluation = conll18_ud_eval.evaluate(gold_ud, system_ud)
	return evaluation['UAS'].f1, evaluation['LAS'].f1

def get_attention_embedding(ids_matrix, last_indexs, config):
	n_ids = last_indexs[-2]
	look_up_words = [0]
	for word_index in range(len(last_indexs) - 2):
		start_index = last_indexs[word_index]
		end_index = last_indexs[word_index + 1]
		look_up_words += [word_index]*(end_index - start_index)

	emb = []
	for word_index in range(len(last_indexs) - 2):
		start_index = last_indexs[word_index]
		end_index = last_indexs[word_index + 1]
		first_score = second_score = first_position = second_position = -1
		for id_index in range(start_index, end_index):
			for j in range(1, n_ids):
				score = ids_matrix[id_index][j]
				position = look_up_words[j]
				if position != word_index and score > second_score:
					if score > first_score:
						second_score = first_score
						second_position = first_position
						first_score = score
						first_position = position
					else:
						second_score = score
						second_position = position
		emb.append([first_position, second_position])
	return emb


def is_required_attention_head(layer_index, head_index, requires):
	for layer, head in requires:
		if layer == layer_index and (head == head_index or head == '*'):
			return True
	return False

# attention format: [layer: 12][batch][head: 12][ids][ids]
# attention requires format: [(a,b), (a,b)] with a is hidden layer, b is head, if b is '*' = get all
# output: [batch][words][attention_emb_dim]
def get_attention_heads(attention_heads, config, last_indexs):
	n_layer = 12
	n_heads = 12
	embedding_list = []
	for sentence_index in range(len(last_indexs)):
		embedding = [[] for i in range(len(last_indexs[sentence_index]) - 2)]
		for layer_index in range(n_layer):
			for head_index in range(n_heads):
				if is_required_attention_head(layer_index, head_index, config.attention_requires):
					tmp_emb = get_attention_embedding(attention_heads[layer_index][sentence_index][head_index], last_indexs[sentence_index], config)
					embedding = [current+tmp for current, tmp in zip(embedding, tmp_emb)]
		embedding_list.append(embedding)
	return embedding_list
