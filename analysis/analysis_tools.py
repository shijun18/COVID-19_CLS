import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve,auc
from sklearn.metrics import confusion_matrix
import torch

def plot_confusion_matrix(cm,labels_name,title,save_path):
  cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] 
  plt.figure(figsize=(8,8))
  plt.imshow(cm, interpolation='nearest',cmap=plt.cm.Blues)   
  plt.title(title,fontsize='xx-large')    
  plt.colorbar()
  indices = np.array(range(len(labels_name)))    
  plt.xticks(indices, labels_name, fontsize='xx-large')    
  plt.yticks(indices, labels_name, fontsize='xx-large')                
  plt.ylabel('True label',fontsize='xx-large')    
  plt.xlabel('Predicted label',fontsize='xx-large')
  for first_index in range(len(cm)):    
      for second_index in range(len(cm[first_index])):    
          percent =  cm[first_index][second_index]*100
          color = 'black'if percent < 50 else 'w'
          plt.text(first_index, second_index, '%.2f%%'%(cm[first_index][second_index]*100),color=color,
                  verticalalignment='center',horizontalalignment='center',fontsize='xx-large')
  plt.savefig('{}/{}.png'.format(save_path,title.replace(' ','_')), format='png')
  plt.show()


def plot_roc_curve(true,prob,labels_name,title,save_path):  
  color_list = ['']
  plt.figure(figsize=(8,8))
  lw = 4
  plt.figure(figsize=(10,10))
  for index in range(len(labels_name)):
      fpr,tpr,threshold = roc_curve(y_true=true,y_score=prob[:,index],pos_label=index) 
      roc_auc = auc(fpr,tpr) 
      plt.plot(fpr, tpr, 
                lw=lw, label='%s(AUC = %0.3f)' % (labels_name[index],roc_auc)) 
      plt.plot([0, 1], [0, 1], color='gray', lw=lw, linestyle='--')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('1 - Specificity',fontsize='xx-large')
  plt.ylabel('Sensitivity',fontsize='xx-large')
  plt.title(title,fontsize='xx-large')
  plt.legend(loc="lower right",fontsize='xx-large')
  plt.savefig('{}/{}.png'.format(save_path,title.replace(' ','_')), format='png')
  plt.show()


def get_fc_weight(net,weight_path):
  checkpoint = torch.load(weight_path)
  net.load_state_dict(checkpoint['state_dict']) 
  fc_weight = net.fc.weight.detach().numpy()
  # print(type(fc_weight))
  return fc_weight