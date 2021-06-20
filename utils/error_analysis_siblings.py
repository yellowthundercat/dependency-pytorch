# RUN: python3 error_analysis_siblings

from error_analysis_utils import *
from collections import defaultdict

MAXIMUM_NMS = 10

# read data
error_file = 'data/error_sample_gold_87.16_81.77.txt'
sentence_lst = read_data(error_file)
# example data
# sentence_lst = [sentence_lst[0]]
# print(sentence_lst)
# {'index': [0, '1', '2'], 'word': ['<root>', 'Cảnh_báo', '!'], 'pos': ['ROOT', 'v', '.'], 'system_head': [0, '0', '1'], 'gold_head': [0, '0', '1'], 'system_label': ['0', 'root', 'punct'], 'gold_label': ['0', 'root', 'punct']}

total_word = 0
gold = defaultdict(lambda: 0)  # {'x': '# words having x siblings on GOLD graph'}
pred = defaultdict(lambda: 0)  # {'x': '# words having x siblings on PREDICT graph'}
correct_precision = defaultdict(lambda: 0)
correct_recall = defaultdict(lambda: 0)

for sen in sentence_lst:
  gold_dependents = {}  # 'word': '#dependents of word'
  pred_dependents = {}
  for i in range(0, len(sen['index'])):
    gold_dependents[sen['index'][i]] = 0
    pred_dependents[sen['index'][i]] = 0

  for i in range(1, len(sen['index'])):
    # for each word i, find its head and count number of dependents of the head on gold & predict graph
    gold_head_idx = int(sen['gold_head'][i])
    system_head_idx = int(sen['system_head'][i])
    gold_label_idx = sen['gold_label'][i]
    system_label_idx = sen['system_label'][i]

    gold_dependents[sen['index'][gold_head_idx]] += 1
    pred_dependents[sen['index'][system_head_idx]] += 1

  for i in range(1, len(sen['index'])):
    gold_head_idx = int(sen['gold_head'][i])
    system_head_idx = int(sen['system_head'][i])
    gold_label_idx = sen['gold_label'][i]
    system_label_idx = sen['system_label'][i]

    gold_sib = min(gold_dependents[sen['index'][gold_head_idx]] - 1, MAXIMUM_NMS)
    pre_sib = min(pred_dependents[sen['index'][system_head_idx]] - 1, MAXIMUM_NMS)
    if gold_sib < 0 or pre_sib < 0:
      raise ValueError('wrong compute siblings')

    gold[gold_sib] += 1
    pred[pre_sib] += 1
    total_word += 1

    if gold_head_idx == system_head_idx and gold_label_idx == system_label_idx:
      correct_recall[gold_sib] += 1
      correct_precision[pre_sib] += 1

precision = {}
recall = {}
percent = {}
for pk in range(MAXIMUM_NMS+1):
    precision[pk] = round(correct_precision[pk] / pred[pk] * 100.0, 2) if pred[pk] != 0 else 0
    recall[pk] = round(correct_recall[pk] / gold[pk] * 100.0, 2) if gold[pk] != 0 else 0
    percent[pk] = round(gold[pk] / total_word * 100.0, 2)

print("pred:", pred)
print("gold:", gold)
print("correct_precision:", correct_precision)
print("correct_recall:", correct_recall)
print("precision:", precision)
print("recall:", recall)
print("percentage:", percent)
