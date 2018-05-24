# coding: utf-8

import pandas as pd
import os
import re
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

tags = ['o','b', 'jx', 'jc', 'm', 'qt','w', 's', 't', 'x', 'yc', 'yw', 'zs', 'zl']
tag2label = dict(zip(tags,range(len(tags))))
#{'b': 1, 'bw': 13, 'j': 4, 'jc': 7, 'o': 0, 'qt': 12, 's': 3, 'sb': 11, 't': 8, 'tz': 6, 'x': 2, 'yw': 9, 'zl': 10, 'zs': 5}
				   
def removeWhiteSpace(dir,file):
	"""
	去除每行文本中的空格
	"""
    with open('{}{}.csv'.format(dir,file),'r',encoding='utf8') as f:
        l = f.readlines()
    with open('{}r{}.csv'.format(dir,file),'w',encoding='utf8') as f:
        for i in range(len(l)):
            if i>0:
                l[i]=re.sub(r'[ ]+','',l[i])
        f.writelines(l)
    return 'r{}{}.csv'.format(dir,file)

def splitText(dir,file,fieldlist):
	"""
	把大文件转化为20行的小文件
	"""
    df = pd.read_csv('{}{}.csv'.format(dir,file),encoding='utf8')
    if not os.path.exists('{}{}'.format(dir,file)): 
        os.mkdir('{}{}'.format(dir,file))
    count = 0
    while count*20<df.shape[0]:
        df[fieldlist].iloc[count*20:(count*20+20),:].to_csv('{0}{1}\\{1}_{2}.csv'.format(dir,file,count),index=False,header=False,sep='\n')
        count += 1


def buildCorpus(sourcepath, destpath):
	"""
	把标注文件转换为训练格式
	sourcepath：'d:\\github\\zh-NER-TF\\data_path\\original\\testright1.txt'

	"""
	f=open(sourcepath,'r',encoding='utf8')
	l=f.readlines()
    for i in range(len(l)):
        for k in tag2label.keys():
            l[i] = re.sub('/'+k,'/'+k+' ',l[i])
    f.close()
	g = []
	for ll in [s.split().strip() for s in l]:
        
		gg = []
		for s in ll:
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


if __name__=='__main__':
	dir = 'd:\\Desktop\\cndata\\'
	file = 'u_tb_yl_mz_medical_record'
	removeWhiteSpace(dir, file)
	splitText(dir,'r'+file,['ZS','ZZMS'])
	file = 'u_tb_cis_mzdzbl'
	removeWhiteSpace(dir, file)
	splitText(dir,'r'+file,['ZS','XBS','JWS','TGJC','FZJC'])
	file = 'u_tb_cis_leavehospital_summary'
	removeWhiteSpace(dir, file)
	splitText(dir,'r'+file,['RYZZTZ','JCHZ','TSJC'])