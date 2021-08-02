from utils import utils

input_file_system = 'analysis/parsing_phonlp.txt'
input_file_gold = 'analysis/test.txt'
tmp_system = 'analysis/tmp.txt'
tmp_gold = 'analysis/tmp_gold.txt'

bucket = [10, 20, 30, 40, 50]

def read_sentence(source_file):
    sentences = []
    sentence = []
    for line in source_file:
        if line.startswith('#'):  # skip comment line
            continue
        line = line.strip()
        if line == '' or line == '\n':
            if len(sentence) > 0:
                sentences.append(sentence)
            sentence = []
        else:
            sentence.append(line)
    return sentences

def write_file(sentences, end_position, out_file):
    for sentence in sentences[:end_position]:
        for line in sentence:
            out_line = line.replace('LBKT', '(')
            out_line = out_line.replace('RBKT', ')')
            out_file.write(out_line)
            out_file.write('\n')
        out_file.write('\n')

def calculate(maximum, sys, gold):
    out_system = open(tmp_system, 'w', encoding='utf-8')
    out_gold = open(tmp_gold, 'w', encoding='utf-8')
    if maximum == -1:
        end_position = len(sys)
    else:
        end_position = 0
        for i, sentence in enumerate(sys):
            if len(sentence) <= maximum:
                end_position = i + 1
    write_file(sys, end_position, out_system)
    write_file(gold, end_position, out_gold)
    out_system.close()
    out_gold.close()
    uas, las = utils.ud_scores(tmp_gold, tmp_system)
    # print(f'bucket {maximum} {uas*100:.2f} {las*100:.2f}')
    print(f'{las*100:.2f}')
    return sys[end_position:], gold[end_position:]

in_system = open(input_file_system, encoding='utf-8')
in_gold = open(input_file_gold, encoding='utf-8')
system_sentence = read_sentence(in_system)
system_sentence.sort(key=lambda x: len(x))
gold_sentence = read_sentence(in_gold)
gold_sentence.sort(key= lambda x: len(x))
for length in bucket:
    system_sentence, gold_sentence = calculate(length, system_sentence, gold_sentence)
system_sentence, gold_sentence = calculate(-1, system_sentence, gold_sentence)
