import sys, pickle, os, random
import numpy as np
import datapreprocessing as dp

# l = []
# l.append('{}'.format('O'))
# for k in dp.tag2label.keys():
#     if k != 'o':
#         l.append('B-{}'.format(k.upper()))
#         l.append('I-{}'.format(k.upper()))
        
# tag2label = dict(zip(l,range(len(l))))        
# ## tags, BIO
# tag2label = {"O": 0,
#              "B-PER": 1, "I-PER": 2,
#              "B-LOC": 3, "I-LOC": 4,
#              "B-ORG": 5, "I-ORG": 6
#              }
tag2label = {'B-1B': 27,
 'B-1JC': 55,
 'B-1JX': 41,
 'B-1M': 69,
 'B-1QT': 83,
 'B-1S': 111,
 'B-1T': 125,
 'B-1W': 97,
 'B-1X': 139,
 'B-1YC': 153,
 'B-1YW': 167,
 'B-1ZL': 195,
 'B-1ZS': 181,
 'B-2B': 29,
 'B-2JC': 57,
 'B-2JX': 43,
 'B-2M': 71,
 'B-2QT': 85,
 'B-2S': 113,
 'B-2T': 127,
 'B-2W': 99,
 'B-2X': 141,
 'B-2YC': 155,
 'B-2YW': 169,
 'B-2ZL': 197,
 'B-2ZS': 183,
 'B-3B': 31,
 'B-3JC': 59,
 'B-3JX': 45,
 'B-3M': 73,
 'B-3QT': 87,
 'B-3S': 115,
 'B-3T': 129,
 'B-3W': 101,
 'B-3X': 143,
 'B-3YC': 157,
 'B-3YW': 171,
 'B-3ZL': 199,
 'B-3ZS': 185,
 'B-4B': 33,
 'B-4JC': 61,
 'B-4JX': 47,
 'B-4M': 75,
 'B-4QT': 89,
 'B-4S': 117,
 'B-4T': 131,
 'B-4W': 103,
 'B-4X': 145,
 'B-4YC': 159,
 'B-4YW': 173,
 'B-4ZL': 201,
 'B-4ZS': 187,
 'B-5B': 35,
 'B-5JC': 63,
 'B-5JX': 49,
 'B-5M': 77,
 'B-5QT': 91,
 'B-5S': 119,
 'B-5T': 133,
 'B-5W': 105,
 'B-5X': 147,
 'B-5YC': 161,
 'B-5YW': 175,
 'B-5ZL': 203,
 'B-5ZS': 189,
 'B-6B': 37,
 'B-6JC': 65,
 'B-6JX': 51,
 'B-6M': 79,
 'B-6QT': 93,
 'B-6S': 121,
 'B-6T': 135,
 'B-6W': 107,
 'B-6X': 149,
 'B-6YC': 163,
 'B-6YW': 177,
 'B-6ZL': 205,
 'B-6ZS': 191,
 'B-7B': 39,
 'B-7JC': 67,
 'B-7JX': 53,
 'B-7M': 81,
 'B-7QT': 95,
 'B-7S': 123,
 'B-7T': 137,
 'B-7W': 109,
 'B-7X': 151,
 'B-7YC': 165,
 'B-7YW': 179,
 'B-7ZL': 207,
 'B-7ZS': 193,
 'B-B': 1,
 'B-JC': 5,
 'B-JX': 3,
 'B-M': 7,
 'B-QT': 9,
 'B-S': 13,
 'B-T': 15,
 'B-W': 11,
 'B-X': 17,
 'B-YC': 19,
 'B-YW': 21,
 'B-ZL': 25,
 'B-ZS': 23,
 'I-1B': 28,
 'I-1JC': 56,
 'I-1JX': 42,
 'I-1M': 70,
 'I-1QT': 84,
 'I-1S': 112,
 'I-1T': 126,
 'I-1W': 98,
 'I-1X': 140,
 'I-1YC': 154,
 'I-1YW': 168,
 'I-1ZL': 196,
 'I-1ZS': 182,
 'I-2B': 30,
 'I-2JC': 58,
 'I-2JX': 44,
 'I-2M': 72,
 'I-2QT': 86,
 'I-2S': 114,
 'I-2T': 128,
 'I-2W': 100,
 'I-2X': 142,
 'I-2YC': 156,
 'I-2YW': 170,
 'I-2ZL': 198,
 'I-2ZS': 184,
 'I-3B': 32,
 'I-3JC': 60,
 'I-3JX': 46,
 'I-3M': 74,
 'I-3QT': 88,
 'I-3S': 116,
 'I-3T': 130,
 'I-3W': 102,
 'I-3X': 144,
 'I-3YC': 158,
 'I-3YW': 172,
 'I-3ZL': 200,
 'I-3ZS': 186,
 'I-4B': 34,
 'I-4JC': 62,
 'I-4JX': 48,
 'I-4M': 76,
 'I-4QT': 90,
 'I-4S': 118,
 'I-4T': 132,
 'I-4W': 104,
 'I-4X': 146,
 'I-4YC': 160,
 'I-4YW': 174,
 'I-4ZL': 202,
 'I-4ZS': 188,
 'I-5B': 36,
 'I-5JC': 64,
 'I-5JX': 50,
 'I-5M': 78,
 'I-5QT': 92,
 'I-5S': 120,
 'I-5T': 134,
 'I-5W': 106,
 'I-5X': 148,
 'I-5YC': 162,
 'I-5YW': 176,
 'I-5ZL': 204,
 'I-5ZS': 190,
 'I-6B': 38,
 'I-6JC': 66,
 'I-6JX': 52,
 'I-6M': 80,
 'I-6QT': 94,
 'I-6S': 122,
 'I-6T': 136,
 'I-6W': 108,
 'I-6X': 150,
 'I-6YC': 164,
 'I-6YW': 178,
 'I-6ZL': 206,
 'I-6ZS': 192,
 'I-7B': 40,
 'I-7JC': 68,
 'I-7JX': 54,
 'I-7M': 82,
 'I-7QT': 96,
 'I-7S': 124,
 'I-7T': 138,
 'I-7W': 110,
 'I-7X': 152,
 'I-7YC': 166,
 'I-7YW': 180,
 'I-7ZL': 208,
 'I-7ZS': 194,
 'I-B': 2,
 'I-JC': 6,
 'I-JX': 4,
 'I-M': 8,
 'I-QT': 10,
 'I-S': 14,
 'I-T': 16,
 'I-W': 12,
 'I-X': 18,
 'I-YC': 20,
 'I-YW': 22,
 'I-ZL': 26,
 'I-ZS': 24,
 'O': 0}

def read_corpus(corpus_path):
    """
    read corpus and return the list of samples
    :param corpus_path:
    :return: data
    """
    data = []
    with open(corpus_path, encoding='utf-8') as fr:
        lines = fr.readlines()
    sent_, tag_ = [], []
    for line in lines:
        if line != '\n':
            [char, label] = line.strip().split()
            sent_.append(char)
            tag_.append(label)
        else:
            data.append((sent_, tag_))
            sent_, tag_ = [], []

    return data


def vocab_build(vocab_path, corpus_path, min_count):
    """

    :param vocab_path:
    :param corpus_path:
    :param min_count:
    :return:
    """
    data = read_corpus(corpus_path)
    word2id = {}
    for sent_, tag_ in data:
        for word in sent_:
            if word.isdigit():
                word = '<NUM>'
            elif ('\u0041' <= word <='\u005a') or ('\u0061' <= word <='\u007a'):
                word = '<ENG>'
            if word not in word2id:
                word2id[word] = [len(word2id)+1, 1]
            else:
                word2id[word][1] += 1
    #删除低频字
    low_freq_words = []
    for word, [word_id, word_freq] in word2id.items():
        if word_freq < min_count and word != '<NUM>' and word != '<ENG>':
            low_freq_words.append(word)
    for word in low_freq_words:
        del word2id[word]
    #为每个字编码
    new_id = 1
    for word in word2id.keys():
        word2id[word] = new_id
        new_id += 1
    word2id['<UNK>'] = new_id
    word2id['<PAD>'] = 0

    print(len(word2id))
    with open(vocab_path, 'wb') as fw:
        pickle.dump(word2id, fw)


def sentence2id(sent, word2id):
    """

    :param sent:
    :param word2id:
    :return:
    """
    sentence_id = []
    for word in sent:
        if word.isdigit():
            word = '<NUM>'
        #判断word是不是english，注意这里只比较word的首字母是否在A-Z 或 a-z
        elif ('\u0041' <= word <= '\u005a') or ('\u0061' <= word <= '\u007a'):
            word = '<ENG>'
        if word not in word2id:
            word = '<UNK>'
        sentence_id.append(word2id[word])
    return sentence_id


def read_dictionary(vocab_path):
    """

    :param vocab_path:
    :return:
    """
    vocab_path = os.path.join(vocab_path)
    with open(vocab_path, 'rb') as fr:
        word2id = pickle.load(fr)
    print('vocab_size:', len(word2id))
    return word2id


def random_embedding(vocab, embedding_dim):
    """

    :param vocab:
    :param embedding_dim:
    :return:
    """
    embedding_mat = np.random.uniform(-0.25, 0.25, (len(vocab), embedding_dim))
    embedding_mat = np.float32(embedding_mat)
    return embedding_mat


def pad_sequences(sequences, pad_mark=0):
    """

    :param sequences:
    :param pad_mark:
    :return:
    """
    max_len = max(map(lambda x : len(x), sequences))
    seq_list, seq_len_list = [], []
    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_len] + [pad_mark] * max(max_len - len(seq), 0)
        seq_list.append(seq_)
        seq_len_list.append(min(len(seq), max_len))
    return seq_list, seq_len_list


def batch_yield(data, batch_size, vocab, tag2label, shuffle=False):
    """

    :param data:
    :param batch_size:
    :param vocab:
    :param tag2label:
    :param shuffle:
    :return:
    """
    if shuffle:
        random.shuffle(data)

    seqs, labels = [], []
    for (sent_, tag_) in data:
        sent_ = sentence2id(sent_, vocab)
        label_ = [tag2label[tag] for tag in tag_]

        if len(seqs) == batch_size:
            yield seqs, labels
            seqs, labels = [], []

        seqs.append(sent_)
        labels.append(label_)

    if len(seqs) != 0:
        yield seqs, labels

