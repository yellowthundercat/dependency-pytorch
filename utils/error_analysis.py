# RUN: python3 error_analysis_siblings

from error_analysis_utils import *
from collections import defaultdict

MAXIMUM_NMS = 10

# read data
error_file = '/Users/trinh.ngo/Documents/NLP/code/dependency-pytorch/data/error_sample_gold_87.16_81.77.txt'
sentence_lst = read_data(error_file)
# example data
# sentence_lst = [sentence_lst[0]]
# print(sentence_lst)
# {'word': [0, '1', '2'],
# 'word': ['<root>', 'Cảnh_báo', '!'],
# 'pos': ['ROOT', 'v', '.'],
# 'system_head': [0, '0', '1'],
# 'gold_head': [0, '0', '1'],
# 'system_label': ['0', 'root', 'punct'],
# 'gold_label': ['0', 'root', 'punct']
# }

def error_analysis_siblings(sentence_lst):
  total_word = 0
  gold = defaultdict(lambda: 0) # {'x (number of modifier siblings (NMS))': '# words having x siblings on gold graph, default = 0'}
  pred = defaultdict(lambda: 0) # {'x': '# words having x siblings on predict graph, default = 0'}
  correct_recall = defaultdict(lambda: 0) # {'x': '# words having x siblings on gold graph having correct prediction, default = 0'}
  correct_precision = defaultdict(lambda: 0) # {'x': '# words having x siblings on predict graph having correct prediction, default = 0'}

  for sen in sentence_lst:
    gold_dependents = defaultdict(lambda: 0)  # 'word': '#dependents of word in the sentence on gold graph'
    pred_dependents = defaultdict(lambda: 0)  # 'word': '#dependents of word in the sentence on predict graph'

    for i in range(1, len(sen['word'])):
      # for the word i, find its gold & predict HEAD
      gold_head_idx = int(sen['gold_head'][i])
      system_head_idx = int(sen['system_head'][i])

      # count number of dependents of the HEAD on gold & predict graph
      gold_dependents[sen['word'][gold_head_idx]] += 1
      pred_dependents[sen['word'][system_head_idx]] += 1

    for i in range(1, len(sen['word'])):
      print("WORD:", sen['word'][i])
      # for the word i, find its gold & predict HEAD
      gold_head_idx = int(sen['gold_head'][i])
      system_head_idx = int(sen['system_head'][i])
      gold_label_idx = sen['gold_label'][i]
      system_label_idx = sen['system_label'][i]

      # count number of siblings of the HEAD on gold & predict graph, if it > MAXIMUM_NMS then it equals MAXIMUM_NMS
      gold_sib = min(gold_dependents[sen['word'][gold_head_idx]] - 1, MAXIMUM_NMS)
      pre_sib = min(pred_dependents[sen['word'][system_head_idx]] - 1, MAXIMUM_NMS)
      if gold_sib < 0 or pre_sib < 0:
        raise ValueError('wrong compute siblings')

      # increase number of words having x siblings on gold & predict graph, x from 0 -> MAXIMUM_NMS
      gold[gold_sib] += 1
      pred[pre_sib] += 1
      total_word += 1

      # if HEAD of i having correct prediction for both arc & label, increase number of words having x siblings on correct gold & predict graph
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

  print("pred:\n", pred)
  print("gold:\n", gold)
  print("correct_precision:\n", correct_precision)
  print("correct_recall:\n", correct_recall)
  print("precision:\n", precision)
  print("recall:\n", recall)
  print("percentage:\n", percent)

# ----------------------------#

def is_descendant(u, w, sen):
  # check if u is dependent of w, condition is u!= root because an arbitrary word always trace back to root
  cnt_arc = 0
  while (u > 0):
    if u == w:
      return True
    gold_head_idx = int(sen['gold_head'][u])
    u = gold_head_idx
    cnt_arc += 1
    # print(cnt_arc)
  return False

def is_right_word(i, sen, head_name):
  # for the word i, find its gold & predict HEAD
  head_i = int(sen[head_name][i])
  check_1 = check_2 = check_3 = False

  for idx in range(min(i, head_i), max(i, head_i)):
    # the right word idx must NOT be the descendants of head of i
    if not is_descendant(idx, head_i, sen):
      check_1 = True

    # the right word idx must modify a word occur outside the interval [i, head of i]
    omax = max(i, head_i) + 1
    omin = min(i, head_i) - 1
    while omin > 0:
      gold_head_o = int(sen[head_name][omin])
      if gold_head_o == idx:
        check_2 = True
      omin -= 1
    while omax < len(sen['word']):
      gold_head_o = int(sen[head_name][omax])
      if gold_head_o == idx:
        check_3 = True
      omax += 1
  if check_1 and (check_2 or check_3):
    return True

def error_analysis_non_projective_arc_degree(sentence_lst):
  for sen in sentence_lst:
    cnt_right_word = 0
    for i in range(1, len(sen['word'])):
      if is_right_word(i, sen, 'system_head'):
        cnt_right_word += 1
      # print(i, cnt_right_word)
    if cnt_right_word > 0:
      print(cnt_right_word)
      print(sen, '\n')
    # break

error_analysis_siblings(sentence_lst)
error_analysis_non_projective_arc_degree(sentence_lst)