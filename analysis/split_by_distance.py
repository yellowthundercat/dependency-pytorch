from collections import defaultdict

input_file_name = 'analysis/error_sample_phonlp.txt'

correct_dict = defaultdict(lambda: 0)
correct_label_dict = defaultdict(lambda: 0)
predict_dict = defaultdict(lambda: 0)
gold_dict = defaultdict(lambda: 0)
input_file = open(input_file_name, encoding='utf-8')
for index, line in enumerate(input_file):
    if line.startswith('#') or line == '' or line == '\n' or index == 0:  # skip comment line + format file
        continue
    token = line.split('\t')
    current_position = int(token[0])
    sys_position = int(token[3])
    gold_position = int(token[4])
    sys_dis = max(min(current_position-sys_position, 6), -6)
    gold_dis = max(min(current_position-gold_position, 6), -6)
    predict_dict[sys_dis] += 1
    gold_dict[gold_dis] += 1
    if sys_position == gold_position:
        correct_dict[sys_dis] += 1
        if token[5] + '\n' == token[6]:
            correct_label_dict[sys_dis] += 1

for i in range(-6, 7):
    if i != 0:
        precision = correct_dict[i] / predict_dict[i]
        recall = correct_dict[i] / gold_dict[i]
        f1 = 2.0 * (precision*recall) / (precision+recall)
        # print(f'distance unlabeled {i} {f1*100:.2f}')
        print(f'{f1 * 100:.2f}')

print('-'*20)
for i in range(-6, 7):
    if i != 0:
        precision = correct_label_dict[i] / predict_dict[i]
        recall = correct_label_dict[i] / gold_dict[i]
        if precision*recall != 0:
            f1 = 2.0 * (precision * recall) / (precision + recall)
        else:
            f1 = 0
        # print(f'distance labeled {i} {f1 * 100:.2f}')
        print(f'{f1 * 100:.2f}')

