import os

# system than gold
def write_file_error_example(config, vocab, words, sys_heads, gold_heads, sys_labels, gold_labels, lengths):
	output_file = open(config.error_sample_file, 'w', encoding='utf-8')
	order = []
	for index, lents in enumerate(lengths):
		token_correct = 0
		for sys_head, sys_lab, gold_head, gold_lab in zip(sys_heads[index], sys_labels[index], gold_heads[index], gold_labels[index]):
			if sys_head == gold_head:
				token_correct += 1
			if sys_lab == gold_lab:
				token_correct += 1
		order.append((index, (token_correct+1)/(2*lents+1)))
	new_order, _ = zip(*sorted(order, key=lambda t: t[1]))
	for index in new_order:
		word_index = 0
		for word, sys_head, sys_lab, gold_head, gold_lab in zip(words[index], sys_heads[index], sys_labels[index], gold_heads[index], gold_labels[index]):
			if word_index > 0:
				output_file.write(f'{word_index}\t{word}\t{sys_head}\t{gold_head}\t{vocab.i2l[sys_lab]}\t{vocab.i2l[gold_lab]}\n')
			word_index += 1
		output_file.write('\n')
