# RUN: python3 error_analysis_siblings

from error_analysis_utils import *

# read data
error_file = 'error_sample_gold_87.16_81.77.txt'
sentence_lst = read_data(error_file)
# example data
# sentence_lst = [sentence_lst[0]]
# print(sentence_lst)
# {'index': [0, '1', '2'], 'word': ['<root>', 'Cảnh_báo', '!'], 'pos': ['ROOT', 'v', '.'], 'system_head': [0, '0', '1'], 'gold_head': [0, '0', '1'], 'system_label': ['0', 'root', 'punct'], 'gold_label': ['0', 'root', 'punct']}

gold = {} # {'x': '# words having x siblings on GOLD graph'}
pred = {} # {'x': '# words having x siblings on PREDICT graph'}
correct_precision = {}
correct_recall = {}

for sen in sentence_lst:
  gold_dependents = {} # 'word': '#dependents of word'
  pred_dependents = {}
  correct_gold_dependents = {}
  correct_pred_dependents = {}
  gold_siblings = {} # 'word': '#siblings of word'
  pred_siblings = {}
  correct_gold_siblings = {}
  correct_pred_siblings = {}
  for i in range(0, len(sen['index'])):
    gold_dependents[sen['word'][i]] = 0
    pred_dependents[sen['word'][i]] = 0
    correct_gold_dependents[sen['word'][i]] = 0
    correct_pred_dependents[sen['word'][i]] = 0
    gold_siblings[sen['word'][i]] = 0
    pred_siblings[sen['word'][i]] = 0
    correct_gold_siblings[sen['word'][i]] = 0
    correct_pred_siblings[sen['word'][i]] = 0

  for i in range(1, len(sen['index'])):
    # for each word i, find its head and count number of dependents of the head on gold & predict graph
    gold_head_idx = int(sen['gold_head'][i])
    system_head_idx = int(sen['system_head'][i])
    gold_label_idx = sen['gold_label'][i]
    system_label_idx = sen['system_label'][i]

    gold_dependents[sen['word'][gold_head_idx]] += 1
    pred_dependents[sen['word'][system_head_idx]] += 1

    # for each word, count number of correct dependents for gold & predict graph
    if gold_head_idx == system_head_idx and gold_label_idx == system_label_idx:
      correct_gold_dependents[sen['word'][gold_head_idx]] += 1
      correct_pred_dependents[sen['word'][system_head_idx]] += 1

  print(gold_dependents, "gold_dependents:")
  # print("pred_dependents:", pred_dependents)
  print(correct_gold_dependents, "correct_gold_dependents:")
  # print("correct_pred_dependents:", correct_pred_dependents)

  for i in range(0, len(sen['index'])):
    gold_head_idx = int(sen['gold_head'][i])
    system_head_idx = int(sen['system_head'][i])

    gold_siblings[sen['word'][i]] = max(0, gold_dependents[sen['word'][gold_head_idx]] - 1)
    pred_siblings[sen['word'][i]] = max(0, pred_dependents[sen['word'][system_head_idx]] - 1)

    correct_gold_siblings[sen['word'][i]] = max(0, correct_gold_dependents[sen['word'][gold_head_idx]] - 1)
    correct_pred_siblings[sen['word'][i]] = max(0, correct_pred_dependents[sen['word'][system_head_idx]] - 1)

  # print(gold_siblings, 'gold_siblings:')
  # print(correct_gold_siblings, 'correct_gold_siblings:')

  update_dict(gold_siblings, gold)
  update_dict(pred_siblings, pred)
  update_dict(correct_pred_siblings, correct_precision)
  update_dict(correct_gold_siblings, correct_recall)

precision = {}
recall = {}
for pk in pred:
  for cp in correct_precision:
    if pk == cp:
      precision[pk] = round(correct_precision[pk] / pred[pk], 2) if pred[pk] != 0 else 0

for gk in gold:
  for cr in correct_recall:
    if gk == cr:
      print(correct_recall[gk])
      print(gold[gk])
      recall[gk] = round(correct_recall[gk] / gold[gk], 2) if gold[gk] != 0 else 0

print("gold:", gold)
print("pred:", pred)
print("correct_precision:", correct_precision)
print("correct_recall:", correct_recall)
print("precision:", precision)
print("recall:", recall)
