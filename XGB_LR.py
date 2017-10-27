#!/usr/bin/env python
#encoding=utf8
'''
  Author: zldeng
  create@2017-09-21 11:23:12
'''
 
import sys,os
reload(sys)
sys.setdefaultencoding('utf8')
 
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression as LR

import xgboost as xgb
import numpy as np
import pickle

from xgboost import XGBClassifier, DMatrix

class XGBoostLR(object):
	'''
	xboost as feature transform.
	xgboost'output is the input feature of LR model
	'''
	def __init__(self,xgb_model_name,lr_model_name,
			one_hot_encoder_model_name,
			xgb_eval_metric = 'mlogloss',xgb_nthread = 32,n_estimators = 100):
		self.xgb_model_name = xgb_model_name
		self.lr_model_name = lr_model_name
		self.one_hot_encoder_model_name = one_hot_encoder_model_name
		self.xgb_eval_metric = xgb_eval_metric
		self.xgb_nthread = xgb_nthread

		self.init_flag = False

	def trainModel(self,train_x,train_y):
		#train a xgboost model
		sys.stdout.flush()
		self.xgb_clf = xgb.XGBClassifier(nthread = self.xgb_nthread)
		self.xgb_clf.fit(train_x,train_y,eval_metric = self.xgb_eval_metric,
				eval_set = [(train_x,train_y)])

		xgb_eval_result = self.xgb_clf.evals_result()
		print 'XGB_train eval_result:',xgb_eval_result
		sys.stdout.flush()

		train_x_mat = DMatrix(train_x)
		print 'get boost tree leaf info...'	
		train_xgb_pred_mat = self.xgb_clf.get_booster().predict(train_x_mat,
				pred_leaf = True)
		print 'get boost tree leaf info done\n'
		
		print 'begin one-hot encoding...'
		self.one_hot_encoder = OneHotEncoder()
		train_lr_feature_mat = self.one_hot_encoder.fit_transform(train_xgb_pred_mat)
		print 'one-hot encoding done!\n\n'
		print 'train_mat:',train_lr_feature_mat.shape
		sys.stdout.flush()
		#train a LR model
		self.lr_clf = LR()
		self.lr_clf.fit(train_lr_feature_mat,train_y)
		
		self.init_flag = True
		
		print 'dump xgboost+lr model..'
		pickle.dump(self.xgb_clf,file(self.xgb_model_name,'wb'),True)
		pickle.dump(self.lr_clf,file(self.lr_model_name,'wb'),True)
		pickle.dump(self.one_hot_encoder,file(self.one_hot_encoder_model_name,'wb'),True)

		print 'Train xgboost and lr model done'
	
	def loadModel(self):
		try:
			self.xgb_clf = pickle.load(file(self.xgb_model_name,'rb'))
			self.lr_clf = pickle.load(file(self.lr_model_name,'rb'))
			self.one_hot_encoder = pickle.load(file(self.one_hot_encoder_model_name,'rb'))

			self.init_flag = True
		except Exception,e:
			print 'Load XGB and LR model fail. + ' + str(e)
			sys.exit(1)
	
	def testModel(self,test_x,test_y):
		if not self.init_flag:
			self.loadModel()
		
		test_x_mat = DMatrix(test_x)
		xgb_pred_mat = self.xgb_clf.get_booster().predict(test_x_mat,pred_leaf = True)
		
		lr_feature = self.one_hot_encoder.transform(xgb_pred_mat)
		#print 'test_mat:',lr_feature.shape

		lr_pred_res = self.lr_clf.predict(lr_feature)

		total = len(test_y)
		correct = 0

		for idx in range(total):
			if lr_pred_res[idx] == test_y[idx]:
				correct += 1

		print 'XGB+LR test: ',total,correct,correct*1.0/total
