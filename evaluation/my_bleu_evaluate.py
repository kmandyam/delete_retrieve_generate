"""
BLEU evaluation script from Li et al. (2018)
Paper: https://arxiv.org/pdf/1804.06437.pdf
Original Code Reference: https://github.com/lijuncen/Sentiment-and-Style-Transfer
Authors: Juncen Li, Robin Jia, He He, Percy Liang
"""

import math
import sys
import os

def load_dict(file_name):
    word_dict = {}
    f = open(file_name, 'r')
    for line in f:
        lines = line.strip().split('\t')
        if (len(lines) == 2):
            word_dict[lines[0]] = int(lines[1])
    return word_dict

def sen_to_array(sen, word_dict, sen1):
    sens = sen.strip().split(' ')
    words = []
    for i in sens:
        if (word_dict.get(i) != None):
            words.append(word_dict.get(i))
    return words

def compute(candidate, references, weights):
    p_ns = (modified_precision(candidate, references, i) for i, _ in enumerate(weights, start=1))

    s = math.fsum(w * math.log(p_n) for w, p_n in zip(weights, p_ns) if p_n)

    bp = brevity_penalty(candidate, references)
    return bp * math.exp(s)

def modified_precision(candidate, references, n):
    counts = counter_gram(candidate, n)

    if not counts:
        return 0
    max_counts = {}
    for reference in references:
        reference_counts = counter_gram(reference, n)
        for ngram in counts.keys():
            if (reference_counts.get(ngram) != None):
                max_counts[ngram] = max(max_counts.get(ngram, 0), reference_counts[ngram])
            else:
                max_counts[ngram] = 0.000000001
    clipped_counts = dict((ngram, min(counts[ngram], max_counts[ngram])) for ngram in counts.keys())

    return sum(clipped_counts.values()) / sum(counts.values())

def counter_gram(word_array, n):
    ngram_words = {}
    for i in range(0, len(word_array) - n + 1):
        tmp_i = ''
        for j in range(0, n):
            tmp_i += str(word_array[i + j])
            tmp_i += ' '
        if (ngram_words.get(tmp_i) == None):
            ngram_words[tmp_i] = 1
        else:
            ngram_words[tmp_i] += 1
    return ngram_words

def brevity_penalty(candidate, references):
    c = len(candidate)
    r = min(abs(len(r) - c) for r in references)
    if c == 0:
        return 0
    if c > r:
        return 1
    else:
        return math.exp(1 - r / c)

if __name__ == "__main__":
    DICT_FILENAME = os.path.join(os.path.dirname(__file__), 'zhi.dict.orgin')
    word_dict = load_dict(DICT_FILENAME)  # dict_file
    can = {}
    query = ''
    answer = []

    weight_num = 4
    weight = []
    for i in range(weight_num):
        weight.append(1.0 / weight_num)

    f = open(sys.argv[1], 'r')  # generate_file
    for line in f:
        lines = line.strip().split('\t')
        if len(lines) == 3:
            can[lines[0].strip()] = sen_to_array(lines[1].strip(), word_dict, lines[0])
    f.close()
    print(len(can))
    ref = {}

    HUMAN_OUTPUTS = os.path.join(os.path.dirname(__file__), 'human.outputs')
    f = open(HUMAN_OUTPUTS, 'r')  # orgin_file
    for line in f:
        lines = line.strip().split('\t')
        if (len(lines) == 2):
            lines[0] = lines[0]
            if (ref.get(lines[0].strip()) == None):
                tmp = []
                tmp.append(sen_to_array(lines[1].strip(), word_dict, lines[0]))
                ref[lines[0].strip()] = tmp
            else:
                ref[lines[0].strip()].append(sen_to_array(lines[1].strip(), word_dict, lines[0]))
    f.close()
    print(len(ref))

    bleu_array = []
    bleu_total = 0
    for i in can.keys():
        if (ref.get(i) != None):
            bleu_score = compute(can[i], ref[i], weight)
            bleu_total += bleu_score
            bleu_array.append(bleu_score)

    print(bleu_total / len(bleu_array))