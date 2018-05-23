# coding: utf-8

import pandas as pd
import os
import pickle
import argparse
##cmd
#python .\datapreprocessing.py -d './tmp/data' formattransform D:\github\zh-NER-TF\data_path\original\testright1.txt
#python .\datapreprocessing.py  -d './tmp/filename' splitfile 'D:\baiduyunpanbackup\datasets\sample-data\sample-data.csv'
## hyperparameters
parser = argparse.ArgumentParser(description='datapreprocessing')
parser.add_argument('mode', type=str, choices=['splitfile','buildcorpus'], help='mode select')
parser.add_argument('sourcepath', type=str, help='source file path')
parser.add_argument('-d','--destpath', type=str, default='.', help='dest file path')
parser.add_argument('-l','--lineNum', type=int, default=20, help='the line number of small file splitted')
args = parser.parse_args()

tag2label = {'o':0, 'b':1,'x':2,'s':3,'j':4,'l':5,'zs':6,'tz':7,'jc':8,'t':9,'yw':10,'zl':11,'sb':12,'qt':13,'bw':14,'ab':15,'fa':16,'pr':17,'con':18,'po':19,'hy':20,'oc':21,'hi':22}
                    
def splitFile2Small(sourcepath, destpath, lineNum):
	#'D:\\baiduyunpanbackup\\datasets\\sample-data\\sample-data.csv'
	df = pd.read_csv(sourcepath,index_col=0)
	
	count = 0
	while count*lineNum<df.shape[0]:
		#'D:\\github\\zh-NER-TF\\data_path\\tt\\t_{}.csv'
		df.iloc[count*lineNum:(count*lineNum+lineNum),:].to_csv('{}_{}'.format(destpath,count))
		count += 1

def buildCorpus(sourcepath, destpath):
	"""
	把标注文件转换为训练格式
	sourcepath：'d:\\github\\zh-NER-TF\\data_path\\original\\testright1.txt'

	"""
	f=open(sourcepath,'r',encoding='utf8')
	l=f.readlines()

	g = []
	for l in [s.split().strip() for s in l]:
		gg = []
		for s in l:
			i = s.rfind('/')
			flag = s[i+1:]
			beginFlag = 'o'
			endFlag = 'o'
			if flag != 'o':
				beginFlag = 'B-{}'.format(flag.upper())
				endFlag = 'I-{}'.format(flag.upper())
			gg.append('{}\t{}\n'.format(s[0],beginFlag))
			for c in s[1:i]:
				gg.append('{}\t{}\n'.format(c,endFlag))
		g.append(gg)
		g.append('\n')

	with open(destpath, 'w', encoding='utf8') as f:
		for gg in g:
			for s in gg:
				f.write(s)

def readCorpus(corpus_path):
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

def buildVocab(vocab_path, corpus_path, min_count):
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

def sentence2Id(sent, word2id):
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


def loadVocab(vocab_path):
    """
    :param vocab_path:
    :return:
    """
    vocab_path = os.path.join(vocab_path)
    with open(vocab_path, 'rb') as fr:
        word2id = pickle.load(fr)
    print('vocab_size:', len(word2id))
    return word2id

if args.mode == 'splitfile':
	splitFile2Small(args.sourcepath, args.destpath, args.lineNum)
elif args.mode == 'buildcorpus':
	buildCorpus(args.sourcepath, args.destpath)
