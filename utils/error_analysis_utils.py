PAD_TOKEN = '<pad>'
PAD_INDEX = 0

UNK_TOKEN = '<unk>'
UNK_INDEX = 1

ROOT_TOKEN = '<root>'
ROOT_TAG = 'ROOT'
ROOT_LABEL = '_root_'
ROOT_INDEX = 2

def get_column_ud(sentence):
  word_idx_lst = [0]
  word_lst = [ROOT_TOKEN]
  pos_lst = [ROOT_TAG]
  system_head_lst = [0]
  gold_head_lst = [0]
  system_label_lst = ['0']
  gold_label_lst = ['0']

  for word in sentence:
    word_idx_lst.append(word[0])
    word_lst.append(word[1])
    pos_lst.append(word[2])
    system_head_lst.append(word[3])
    gold_head_lst.append(word[4])
    system_label_lst.append(word[5])
    gold_label_lst.append(word[6])
  return {'index': word_idx_lst, 'word': word_lst, 'pos': pos_lst, 'system_head': system_head_lst, \
          'gold_head': gold_head_lst, 'system_label': system_label_lst, 'gold_label': gold_label_lst}

def read_data(filename):
  sentence_count = 0
  input_file = open(filename, encoding='utf-8')
  sentence_lst = []
  sentence = []
  lines = next(input_file)
  lines = next(input_file)

  for line in input_file:
    if line.startswith('FORMAT') or line.startswith('#'):
      continue
    line = line.strip()
    if line == '' or line == '\n':
      if len(sentence) > 1:
        sentence_lst.append(get_column_ud(sentence))
        sentence = []
        sentence_count += 1
    else:
      sentence.append(line.split('\t'))
  if len(sentence) > 1:
    sentence_lst.append(get_column_ud(sentence))
  return sentence_lst

def update_dict(dependents, count_dict):
  """
  # word mà cha có x dependents trên cây predict
  """
  for value in dependents.values():
    if value not in count_dict:
      count_dict[value] = 1
    else:
      count_dict[value] += 1

