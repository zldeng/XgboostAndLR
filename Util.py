#!/usr/bin/env python
#encoding=utf8
'''
  Author: zldeng
  create@2017-07-17 16:22:04
'''
 
import sys
reload(sys)
sys.setdefaultencoding('utf8')

def loadData(data_file):
	'''
	将分好词的语料数据转换成sklearn可直接使用的语料
	input:label word1 word2....
	'''
	data_x = []
	data_y = []	

	for line in file(data_file):
		line_list = line.strip().split('\t')

		info_data = ' '.join(line_list[1:])
		tag = line_list[0].strip()

		if '' == tag:
			continue

		data_x.append(info_data)
		data_y.append(tag)

	return data_x,data_y
	
