# Xgboost+LR for text classification
use xgboost + lr model for text classification.  
xgboost is used to be a feature transform for LR

--- 
# 模型简介
## xgboost
xgboost是一个非常高效、应用广泛的GBDT机器学习库，详细信息可参考[xgbbost](https://xgboost.readthedocs.io/en/latest/)  
xgboost提供了高效简洁的python接口，可用于分类、回归任务。在本实验中使用了xgboost的分类接口。

## LR
LR是一个在工业界使用广泛的线性模型，可用于分类、回归任务。  
本实验中使用sklearn机器学习库中的LR模型。sklearn机器学习库可参考[sklearn](http://scikit-learn.org/)

## xgboost+lr
使用xgboost+lr模型融合方法进行分类、回归的思想最初由facebook在广告预测中提出，论文[Practical Lessons from Predicting Clicks on Ads at Facebook](http://quinonero.net/Publications/predicting-clicks-facebook.pdf)  
在该方法中将xgboost作为feature transform，对于每一个样本使用xgboost进行预测，根据xgboost的预测结果每棵回归树的信息构造新的特征向量作为LR模型的输入  
实验结果表明xgboost+lr能取得比单独使用两个模型都好的效果
