#!/usr/bin/env python
#encoding=utf8
'''
  Author: zldeng
  create@2017-09-19 09:14:01
'''
 
 
import sys,os
reload(sys)
sys.setdefaultencoding('utf8')

from Feature import FeatureModel
from Util import loadData
from SKLearnLR import SKLearnLR
from XGBoost import XGBoost
from XGB_LR import XGBoostLR


max_feature_cnt = 4000
feature_max_df = 0.55
feature_min_df = 3
ngram_range = (1,2)

model_path = './model/'

tfidf_model_name = model_path + 'tfidf_feature.model'
best_feature_model_name = model_path + 'best_feature.model'

xgb_model_name = model_path + 'xgboost.model'
sk_lr_model_name = model_path + 'sklearn.lr.model'
xgbLr_xgb_model_name = model_path + 'xgb_lr.xgboost.model'
xgbLr_lr_model_name = model_path + 'xgb_lr.lr.model'
one_hot_encoder_model_name = model_path + 'xgb_lr_one_hot_encoder.model'


def train(train_data_file):
	train_x,train_y = loadData(train_data_file)

	feature_transfor = FeatureModel(tfidf_model_name,best_feature_model_name)

	feature_transfor.fit(max_feature_cnt,feature_max_df,
			feature_min_df,ngram_range,train_x,train_y)

	model_train_x_feature = feature_transfor.transform(train_x)

	#train a single xgboost model
	print 'train a single xgboost model...'
	xgb_clf = XGBoost(xgb_model_name)
	xgb_clf.trainModel(model_train_x_feature,train_y)
	print 'train single xgboost model done!\n\n'
	
	#train a single LR model
	print 'train a single LR model...'
	lr_clf = SKLearnLR(sk_lr_model_name)
	lr_clf.trainModel(model_train_x_feature,train_y)
	print 'train a single LR model done!\n\n'

	#return 
	#train a XGBoost + LR model
	print 'train a xgboost+lr model'
	xgb_lr_clf = XGBoostLR(xgbLr_xgb_model_name,xgbLr_lr_model_name,one_hot_encoder_model_name)
	xgb_lr_clf.trainModel(model_train_x_feature,train_y)
	print 'train xgboost+lr model done\n\n'
	
	print 'Train Done'

def test(test_data_file):
	test_x,test_y = loadData(test_data_file)
	
	feature_transfor = FeatureModel(tfidf_model_name,best_feature_model_name)
	feature_transfor.loadModel()

	model_test_x_feature = feature_transfor.transform(test_x)

	xgb_clf = XGBoost(xgb_model_name)
	xgb_clf.testModel(model_test_x_feature,test_y)
	

	lr_clf = SKLearnLR(sk_lr_model_name)
	lr_clf.testModel(model_test_x_feature,test_y)

	#return
	xgb_lr_clf = XGBoostLR(xgbLr_xgb_model_name,xgbLr_lr_model_name,one_hot_encoder_model_name)
	xgb_lr_clf.testModel(model_test_x_feature,test_y)



if __name__ == '__main__':
	train_data_file = './data/train.data'
	test_data_file = './data/test.data'

	train(train_data_file)

	test(test_data_file)

