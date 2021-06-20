from collections import defaultdict

input_file_name = 'analysis/error_sample_phonlp.txt'

correct_label_dict = defaultdict(lambda: 0)
predict_dict = defaultdict(lambda: 0)
gold_dict = defaultdict(lambda: 0)
length_dict = defaultdict(lambda: 0)
input_file = open(input_file_name, encoding='utf-8')
total_token = 0
for index, line in enumerate(input_file):
    if line.startswith('#') or line == '' or line == '\n' or index == 0:  # skip comment line + format file
        continue
    token = line.split('\t')
    current_position = int(token[0])
    sys_position = int(token[3])
    gold_position = int(token[4])
    sys_label = token[5].rstrip("\n")
    gold_label = token[6].rstrip("\n")
    total_token += 1
    predict_dict[sys_label] += 1
    gold_dict[gold_label] += 1
    length_dict[gold_label] += abs(current_position-gold_position)
    if sys_position == gold_position and sys_label == gold_label:
        correct_label_dict[sys_label] += 1

result_list = []  # label percent precision recall f1 average_length
for key in gold_dict:
    percent = gold_dict[key] / total_token
    if predict_dict[key] == 0:
        precision = 0
    else:
        precision = correct_label_dict[key] / predict_dict[key]
    if gold_dict[key] == 0:
        recall = 0
    else:
        recall = correct_label_dict[key] / gold_dict[key]
    if precision+recall > 0:
        f1 = 2.0 * (precision*recall) / (precision+recall)
    else:
        f1 = 0
    average_length = length_dict[key] / gold_dict[key]
    result_list.append((key, percent, precision, recall, f1, average_length))
    # print(f'distance unlabeled {i} {f1*100:.2f}')

result_list.sort(key=lambda x:x[1],reverse=True)
for item in result_list:
    if item[1] < 0.01:
        break
    print(f'{item[0]} {item[1]*100:.2f} {item[2]*100:.2f} {item[3]*100:.2f} {item[4]*100:.2f} {item[5]:.2f}')

