source_file = open('data/train.txt', 'r', encoding="utf8")
pos_file = open('data/raw_train_POS.txt', 'r', encoding="utf8")
destination_file = open('data/new_train.txt', 'w', encoding="utf8")

sentences = []
sentence = []
sentence_count = 0
for line in source_file:
	if line.startswith('#'):  # skip comment line
		continue
	line = line.strip()
	if line == '' or line == '\n':
		if len(sentence) > 1:
			sentences.append(sentence)
			sentence = []
			sentence_count += 1
	else:
		sentence.append(line)
if len(sentence) > 1:
	sentences.append(sentence)

for index, line in enumerate(pos_file):
	token_list = line.split()
	if len(token_list) != len(sentences[index]):
		print('not match sentence', token_list, sentences[index])
	else:
		for tok_index, token in enumerate(token_list):
			pos = token.split('/')[-1]
			word_part = token[:len(token)-len(pos)-1]
			if word_part not in sentences[index][tok_index]:
				print('not match word', sentences[index][tok_index])
			else:
				sentences[index][tok_index] += '\t' + pos + '\n'
				destination_file.write(sentences[index][tok_index])
		destination_file.write('\n')

