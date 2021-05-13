source_file = open('test_story.txt', 'r', encoding="utf8")
pos_file = open('all_uni.txt', 'r', encoding="utf8")
destination_file = open('data/test_story.txt', 'w', encoding="utf8")

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
		sentence = []
	else:
		sentence.append(line)
if len(sentence) > 1:
	sentences.append(sentence)
pos_sentence = [line for line in pos_file]


def find_match_sentence():
	for p_line in pos_sentence:
		check = True
		token_list = p_line.split()
		if len(token_list) == len(sentences[index]):
			for tok_index, token in enumerate(token_list):
				pos = token.split('/')[-1]
				word_part = token[:len(token) - len(pos) - 1]
				if word_part in sentences[index][tok_index] or (
						word_part == 'LBKT' and '(' in sentences[index][tok_index]) or (
						word_part == 'RBKT' and ')' in sentences[index][tok_index]):
					check = True
				else:
					check = False
					break
		else:
			check = False
		if check:
			return p_line
	return ''


for index in range(len(sentences)):
	line = find_match_sentence()
	token_list = line.split()
	if len(token_list) != len(sentences[index]):
		print('not match sentence', token_list, sentences[index])
	else:
		for tok_index, token in enumerate(token_list):
			pos = token.split('/')[-1]
			word_part = token[:len(token) - len(pos) - 1]
			if word_part in sentences[index][tok_index] or (
					word_part == 'LBKT' and '(' in sentences[index][tok_index]) or (
					word_part == 'RBKT' and ')' in sentences[index][tok_index]):
				sentences[index][tok_index] += '\n'
				destination_file.write(sentences[index][tok_index])
			else:
				print('not match word', sentences[index][tok_index])
		destination_file.write('\n')
