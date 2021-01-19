import os
import pandas as pd 
import pickle
from ml_classifier import ML_Classifier,params_dict

from sklearn.metrics import make_scorer
from sklearn.metrics import roc_curve,auc
from sklearn.metrics import accuracy_score,precision_score,f1_score,recall_score,roc_auc_score
import copy


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
  'scaler_flag':False
}


if __name__ == "__main__":

  cnn_version = 'v11.2' 
  print(cnn_version)
  csv_path = './input_file/{}_score_clincal_info.csv'.format(cnn_version) 
  # csv_path = './input_file/clean_clincal_info.csv'
  df = pd.read_csv(csv_path).iloc[:,2:9]
  # df = pd.read_csv(csv_path).iloc[:,2:]
  # del df['Progression (Days)']
  print(df)
  # clf_name = 'xgboost' 
  # classifier = ML_Classifier(clf_name=clf_name,params=params_dict[clf_name])
  # model = classifier.trainer(df=df,**SETUP_TRAINER)

  for clf_name in _AVAIL_CLF[2:-2]:
    tmp_df = copy.copy(df)
    print('********** %s **********'%clf_name)
    classifier = ML_Classifier(clf_name=clf_name,params=params_dict[clf_name])
    model,pred_,prob_,true_ = classifier.trainer(df=tmp_df,**SETUP_TRAINER)
    test_result = {}
    test_result['pred'] = list(pred_)
    test_result['true'] = list(true_)
    test_result['prob'] = list(prob_[:,1])
    # print(prob_)
    fpr,tpr,threshold = roc_curve(y_true=true_,y_score=prob_[:,1],pos_label=1) 
    roc_auc = auc(fpr,tpr) 
    print(roc_auc)

    csv_file = pd.DataFrame(test_result)
    csv_file.to_csv('./output_file/{}_{}.csv'.format(cnn_version,clf_name.replace(' ','_')),index=False)
  #save model
  # pkl_filename = "./save_model/{}.pkl".format(clf_name.replace(' ','_'))
  # with open(pkl_filename, 'wb') as file:
  #   pickle.dump(model, file)