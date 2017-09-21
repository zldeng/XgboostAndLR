#!/usr/bin/env python
#encoding=utf8
'''
  Author: zldeng
  create@2017-09-21 11:18:03
'''
import sys,os
reload(sys)
sys.setdefaultencoding('utf8')

from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
 
import numpy as np
import pickle



class FeatureModel(object):
	'''
	this class is used to select feature from the raw text	
	'''
	def __init__(self,feature_vec_model_name,best_feature_model_name):
		self.feature_vec_model_name = feature_vec_model_name
		self.best_feature_model_name = best_feature_model_name

		self.init_flag = False

	def fitModelByData(self,x_train,y_train):
		try:
			best_k = self.max_feature_cnt
			vec_max_df = self.feature_max_df
			vec_min_df = self.feature_min_df
			vec_ngram_range = self.ngram_range

			self.tf_vec_ = TfidfVectorizer(ngram_range=vec_ngram_range, 
					min_df=vec_min_df, max_df=vec_max_df)
			
			self.best_ = SelectKBest(chi2, k=best_k)
				
			train_tf_vec = self.tf_vec_.fit_transform(x_train)
			train_best = self.best_.fit_transform(train_tf_vec,y_train)
		except Exception,e:
			print 'Err: ' + str(e)
			sys.exit(1)

	def setFeatureModelPara(self,max_feature_cnt,feature_max_df,
			feature_min_df,ngram_range):
		self.max_feature_cnt = max_feature_cnt
		self.feature_max_df = feature_max_df
		self.feature_min_df = feature_min_df
		self.ngram_range = ngram_range

	def fit(self,max_feature_cnt,feature_max_df,
			feature_min_df,ngram_range,x_train,y_train):

		self.setFeatureModelPara(max_feature_cnt,feature_max_df,
				feature_min_df,ngram_range)

		self.fitModelByData(x_train,y_train)

		pickle.dump(self.tf_vec_,file(self.feature_vec_model_name,'wb'),True)
		pickle.dump(self.best_,file(self.best_feature_model_name,'wb'),True)

	def loadModel(self):
		try:
			self.tf_vec_ = pickle.load(file(self.feature_vec_model_name,'rb'))
			self.best_ = pickle.load(file(self.best_feature_model_name,'rb'))
			self.init_flag = True
		except Exception,e:
			print 'Load feature model fail'
			sys.exit(1)
	
	def transform(self,x_test):
		if not self.init_flag:
			self.loadModel()

		x_vec = self.tf_vec_.transform(x_test)
		x_best = self.best_.transform(x_vec)

		return x_best
	
