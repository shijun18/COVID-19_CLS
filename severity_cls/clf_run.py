import os
import pandas as pd 
import pickle
from ml_classifier import ML_Classifier,params_dict

from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score,precision_score,f1_score,recall_score,roc_auc_score


_AVAIL_CLF = ['lasso','knn','svm','decision tree', \
'random forest','extra trees','bagging','mlp','xgboost','xgboost_rf']

METRICS = {
  'Accuracy':make_scorer(accuracy_score),
  'Recall':make_scorer(recall_score,average='binary',zero_division=0),
  'Precision':make_scorer(precision_score,average='binary',zero_division=0),
  'F1':make_scorer(f1_score,average='binary',zero_division=0),
  'AUC':make_scorer(roc_auc_score)
  }

SETUP_TRAINER = {
  'target_key':'Critical_illness',
  'test_size':0.2,
  'random_state':21,
  'metric':METRICS,
  'k_fold':5,
  'scaler_flag':True
}


if __name__ == "__main__":

  cnn_version = 'v1.1' 
  csv_path = './input_file/{}_score_clincal_info.csv'.format(cnn_version) 
  df = pd.read_csv(csv_path).iloc[:,2:9]
  # del df['Progression (Days)']
  print(df)
  clf_name = 'xgboost' 
  classifier = ML_Classifier(clf_name=clf_name,params=params_dict[clf_name])
  model = classifier.trainer(df=df,**SETUP_TRAINER)
  # for clf_name in _AVAIL_CLF[:-1]:
  #   import copy
  #   tmp_df = copy.copy(df)
  #   print('********** %s **********'%clf_name)
  #   classifier = ML_Classifier(clf_name=clf_name,params=params_dict[clf_name])
  #   model = classifier.trainer(df=tmp_df,**SETUP_TRAINER)
  
  # save model
  # pkl_filename = "./save_model/{}.pkl".format(clf_name.replace(' ','_'))
  # with open(pkl_filename, 'wb') as file:
  #   pickle.dump(model, file)