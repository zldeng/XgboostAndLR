#!/usr/bin/env python
#encoding=utf8
'''
  Author: zldeng
  create@2017-09-21 11:21:59
'''
 
import sys,os
reload(sys)
sys.setdefaultencoding('utf8')
 
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression as LR

class SKLearnLR(object):
	'''
	LR model for text classification	
	'''
	def __init__(self,lr_model_name):
		self.lr_model_name = lr_model_name
		self.init_flag = False
	
	def trainModel(self,train_x,train_y):
		self.clf = LR()
		self.clf.fit(train_x,train_y)
		self.init_flag = True
		pickle.dump(self.clf,file(self.lr_model_name,'wb'),True)

		
	def loadModel(self):
		try:
			self.clf = pickle.load(file(self.lr_model_name,'rb'))
			self.init_flag = True
		
		except Exception,e:
			print 'Load model fail, ' + str(e)
			sys.exit(1)
	
	def testModel(self,test_x,test_y):
		if not self.init_flag:
			self.loadModel()

		pred_y = self.clf.predict(test_x)
		
		total = len(test_y)
		correct = 0
		for idx in range(total):
			if pred_y[idx] == test_y[idx]:
				correct += 1

		print 'Test LR: ',total,correct,correct*1.0/total
