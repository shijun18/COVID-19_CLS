import os
import pandas as pd



def statistics_metric(input_path,result_path,net_name,ver=[1,2,3,4]):
    csv_info = []
    for i in range(5):
        for k,j in enumerate(ver):
            item = []
            report_path = f'v{j}.{i}_report.csv'
            report_path = os.path.join(input_path,report_path)
            version = f'v{j}.{i}'
            item.append(version)
            item.append(net_name[k-1])

            csv_file = pd.read_csv(report_path,index_col=0)
            csv_file = csv_file.drop(labels='support')  # drop `support`
            csv_file = csv_file.drop(labels=['macro avg','weighted avg'],axis=1)
            csv_file.loc['accuracy'] = csv_file['accuracy'].tolist()
            csv_file = csv_file.drop(labels='accuracy',axis=1)
            csv_file = csv_file.T
            print(csv_file)

            for index in csv_file.index:
                item += csv_file.loc[index].tolist()
            csv_info.append(item)

            print(list(csv_file.columns))
            if i == 4:
                columns = ['version','net_name'] + list(csv_file.columns) * 3

    csv_file = pd.DataFrame(data=csv_info,columns=columns)
    csv_file.to_csv(result_path,index=False)


def statistics_auc(input_path,result_path,net_name,labels_name,ver=[1,2,3,4]):
    from sklearn.metrics import roc_curve,auc
    import numpy as np
    csv_info = []
    for i in range(5):
        for k,j in enumerate(ver):
            item = []
            csv_path = f'v{j}.{i}.csv'
            csv_path = os.path.join(input_path,csv_path)
            version = f'v{j}.{i}'
            item.append(version)
            item.append(net_name[k-1])

            file_csv = pd.read_csv(csv_path)
            true_ = np.asarray(file_csv['true'].values.tolist())

            prob_ = file_csv['prob'].values
            prob_ = np.stack([eval(case) for case in prob_],axis=0)

            for index in range(len(labels_name)):
                fpr,tpr,threshold = roc_curve(y_true=true_,y_score=prob_[:,index],pos_label=index) 
                roc_auc = auc(fpr,tpr)
                item.append(roc_auc)
            
            csv_info.append(item)
            
            if i == 4:
                columns = ['version','net_name'] + labels_name
    
    csv_file = pd.DataFrame(data=csv_info,columns=columns)
    csv_file.to_csv(result_path,index=False)




if __name__ == '__main__':
    
    # statistics metric 
    input_path = './new_result'
    result_path = './new_result_metric.csv'
    net_name = ['r3d_18', 'se_r3d_18','da_18','da_se_18']
    # statistics_metric(input_path,result_path,net_name)
    statistics_metric(input_path,result_path,net_name,[1,2,3,4])

    # statistics AUC
    input_path = './new_result'
    result_path = './new_result_auc.csv'
    net_name = ['r3d_18', 'se_r3d_18','da_18','da_se_18']
    labels_name = ['CP','COVID-19','Normal']
    # statistics_auc(input_path,result_path,net_name,labels_name)
    statistics_auc(input_path,result_path,net_name,labels_name,[1,2,3,4])