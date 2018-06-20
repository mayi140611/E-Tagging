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
'''
parser = argparse.ArgumentParser(description='datapreprocessing')
parser.add_argument('mode', type=str, choices=['splitfile','buildcorpus'], help='mode select')
parser.add_argument('sourcepath', type=str, help='source file path')
parser.add_argument('-d','--destpath', type=str, default='.', help='dest file path')
parser.add_argument('-l','--lineNum', type=int, default=20, help='the line number of small file splitted')
args = parser.parse_args()
'''
# print('tagprocessing')
# tags = ['o','b', 'jx', 'jc', 'm', 'qt','w', 's', 't', 'x', 'yc', 'yw', 'zs', 'zl']
# for t in ['o','b', 'jx', 'jc', 'm', 'qt','w', 's', 't', 'x', 'yc', 'yw', 'zs', 'zl']:
# 	for i in range(1,8):
# 		print('{}{}'.format(i,t))
# 		tags.append('{}{}'.format(i,t))
# print('tagprocessing')
#14个标签
# tag2label = dict(zip(tags,range(len(tags))))
# print('tagprocessed')
tag2label = {'1b': 21,
 '1jc': 35,
 '1jx': 28,
 '1m': 42,
 '1o': 14,
 '1qt': 49,
 '1s': 63,
 '1t': 70,
 '1w': 56,
 '1x': 77,
 '1yc': 84,
 '1yw': 91,
 '1zl': 105,
 '1zs': 98,
 '2b': 22,
 '2jc': 36,
 '2jx': 29,
 '2m': 43,
 '2o': 15,
 '2qt': 50,
 '2s': 64,
 '2t': 71,
 '2w': 57,
 '2x': 78,
 '2yc': 85,
 '2yw': 92,
 '2zl': 106,
 '2zs': 99,
 '3b': 23,
 '3jc': 37,
 '3jx': 30,
 '3m': 44,
 '3o': 16,
 '3qt': 51,
 '3s': 65,
 '3t': 72,
 '3w': 58,
 '3x': 79,
 '3yc': 86,
 '3yw': 93,
 '3zl': 107,
 '3zs': 100,
 '4b': 24,
 '4jc': 38,
 '4jx': 31,
 '4m': 45,
 '4o': 17,
 '4qt': 52,
 '4s': 66,
 '4t': 73,
 '4w': 59,
 '4x': 80,
 '4yc': 87,
 '4yw': 94,
 '4zl': 108,
 '4zs': 101,
 '5b': 25,
 '5jc': 39,
 '5jx': 32,
 '5m': 46,
 '5o': 18,
 '5qt': 53,
 '5s': 67,
 '5t': 74,
 '5w': 60,
 '5x': 81,
 '5yc': 88,
 '5yw': 95,
 '5zl': 109,
 '5zs': 102,
 '6b': 26,
 '6jc': 40,
 '6jx': 33,
 '6m': 47,
 '6o': 19,
 '6qt': 54,
 '6s': 68,
 '6t': 75,
 '6w': 61,
 '6x': 82,
 '6yc': 89,
 '6yw': 96,
 '6zl': 110,
 '6zs': 103,
 '7b': 27,
 '7jc': 41,
 '7jx': 34,
 '7m': 48,
 '7o': 20,
 '7qt': 55,
 '7s': 69,
 '7t': 76,
 '7w': 62,
 '7x': 83,
 '7yc': 90,
 '7yw': 97,
 '7zl': 111,
 '7zs': 104,
 'b': 1,
 'jc': 3,
 'jx': 2,
 'm': 4,
 'o': 0,
 'qt': 5,
 's': 7,
 't': 8,
 'w': 6,
 'x': 9,
 'yc': 10,
 'yw': 11,
 'zl': 13,
 'zs': 12}
				   
def removeWhiteSpace(dir,file):
	"""
	人工打标签前的工作：去除每行文本中的空格
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
	人工打标签前的工作：把大文件转化为20行的小文件
	"""
	df = pd.read_csv('{}{}.csv'.format(dir,file),encoding='utf8')
	if not os.path.exists('{}{}'.format(dir,file)): 
		os.mkdir('{}{}'.format(dir,file))
	count = 0
	while count*20<df.shape[0]:
		df[fieldlist].iloc[count*20:(count*20+20),:].to_csv('{0}{1}\\{1}_{2}.csv'.format(dir,file,count),index=False,header=False,sep='\n')
		count += 1

def mergeCSV(corpus_dir):
	'''
	把小的语料文件合并为大的语料文件
	'''
	for s in os.listdir(corpus_dir):
		if s.endswith('.csv'):
			print(os.path.join(corpus_dir,s))
			with open(os.path.join(corpus_dir,s),'r',encoding='utf8') as f:
				l = f.readlines()
			with open(os.path.join(corpus_dir,'merged.csv'),'a',encoding='utf8') as f:
				f.writelines(l)

def buildCorpus(sourcepath, destpath):
	"""
	把标注文件转换为训练格式
	sourcepath：'d:\\github\\zh-NER-TF\\data_path\\original\\testright1.txt'
	:return 标注的词汇列表
	"""
	f=open(sourcepath,'r',encoding='utf8')
	l=f.readlines()
    #在每个标注后面加空格
	for i in range(len(l)):
		for k in tag2label.keys():
			l[i] = re.sub('/'+k,'/'+k+' ',l[i])
	f.close()
	g = []
	d = dict()
	for ll in [s.split() for s in l]:
		gg = []
		for s in ll:
			s = s.strip()
			i = s.rfind('/')
			flag = s[i+1:]
			beginFlag = 'O'
			endFlag = 'O'
			if flag != 'o' and (flag in tag2label):
				beginFlag = 'B-{}'.format(flag.upper())
				endFlag = 'I-{}'.format(flag.upper())
				word = s[:i]
				if flag not in d:
					d[flag] = []
				d[flag].append(word)  
			gg.append('{}\t{}\n'.format(s[0],beginFlag))
			for c in s[1:i]:
				gg.append('{}\t{}\n'.format(c,endFlag))
		g.append(gg)
		g.append('\n')

	with open(destpath, 'w', encoding='utf8') as f:
		for gg in g:
			for s in gg:
				f.write(s)
	return d
    
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
			if char == '。':#按句进行拆分
				data.append((sent_, tag_))
				sent_, tag_ = [], []                
		else:
			if sent_:
				data.append((sent_, tag_))
				sent_, tag_ = [], []

	return data

def buildVocab(vocab_path, corpus_path, min_count=1):
	"""
    为每个字编码
	:param vocab_path:
	:param corpus_path:
	:param min_count:
	:return:
	"""
	data = readCorpus(corpus_path)
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

	print('words size: ',len(word2id))
	with open(vocab_path, 'wb') as fw:
		pickle.dump(word2id, fw)
	return word2id

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
    载入word编码
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