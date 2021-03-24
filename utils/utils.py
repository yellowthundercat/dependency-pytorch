import os
import sys
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

# attention format: [layer: 12][batch][head: 12][ids][ids]
# attention requires format: [(a,b), (a,b)] with a is hidden layer, b is head, if b is '*' = get all
def get_attention_heads(attention_heads, attention_requires, attention_tops):
	n_layer = 12
	n_heads = 12
	embedding = []
	# for layer in range(n_layer):
	# 	for
