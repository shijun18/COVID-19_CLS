import pandas as pd


def csv_reader_single(csv_file,key_col=None,value_col=None):
  '''
  Extracts the specified single column, return a single level dict.
  The value of specified column as the key of dict.

  Args:
  - csv_file: file path
  - key_col: string, specified column as key, the value of the column must be unique. 
  - value_col: string,  specified column as value
  '''
  file_csv = pd.read_csv(csv_file)
  key_list = file_csv[key_col].values.tolist()
  value_list = file_csv[value_col].values.tolist()
  
  target_dict = {}
  for key_item,value_item in zip(key_list,value_list):
    target_dict[key_item] = value_item

  return target_dict



def csv_reader_multi(csv_file,key_col=None,value_col=None):
  '''
  Extracts the specified multi-column, return a two level dict.
  Args:
  - csv_file: file path
  - key_col: string, specified column as key, the value of the column must be unique. 
  - value_col: list of string, multi-column as value. 
  '''
  if isinstance(value_col,list):

    file_csv = pd.read_csv(csv_file)
    key_list = file_csv[key_col].values.tolist()
    value_list = [{} for i in range(len(key_list))]
    for col in value_col:
      for index,item in enumerate(file_csv[col].values.tolist()):
        value_list[index].update({col:item})
    
    target_dict = {}
    for key_item,value_item in zip(key_list,value_list):
      target_dict[key_item] = value_item

    return target_dict

def txt_reader_single(txt_file):
  target_dict = {}
  with open(txt_file,'r') as fp:
    lines = fp.readlines()
    lines = lines[1:]
    for line in lines:
      key = line.split(',')[0]
      target_dict[key] = int(line.split(',')[1])

  return target_dict  

def get_cross_validation_on_patient(path_list, fold_num, current_fold, label_dict):
    import os,random

    print('total scans:%d'%len(path_list))
    tmp_patient_list = [os.path.basename(case).split('_')[0] for case in path_list]
    patient_list = list(set(tmp_patient_list))
    print('total patients:%d'%len(patient_list))
    patient_list.sort(key=tmp_patient_list.index,reverse=True)  

    _len_ = len(patient_list) // fold_num
    train_id = []
    validation_id = []
  

    end_index = current_fold * _len_
    start_index = end_index - _len_

    validation_id.extend(patient_list[start_index:end_index])
    train_id.extend(patient_list[:start_index])
    train_id.extend(patient_list[end_index:_len_*(fold_num-1)])
    test_id = [item for item in patient_list if item not in train_id + validation_id]

    print("Train patient number:", len(train_id),
          "\nVal patient number:", len(validation_id),
          '\nTest patient number:',len(test_id))

    train_path = []
    validation_path = []
    test_path = []

    for case in path_list:
        if os.path.basename(case).split('_')[0] in train_id:
            train_path.append(case)
        elif os.path.basename(case).split('_')[0] in validation_id:
            validation_path.append(case)
        else:
            test_path.append(case)

    random.shuffle(train_path)
    random.shuffle(validation_path)
    print("Train set length:", len(train_path),
          "\nVal set length:", len(validation_path),
          '\nTest set length:',len(test_path))
    
    train_label = [label_dict[case] for case in train_path]
    print('train CP:',train_label.count(0))
    print('train NCP:',train_label.count(1))
    print('train Normal:',train_label.count(2))
    val_label = [label_dict[case] for case in validation_path]
    print('val CP:',val_label.count(0))
    print('val NCP:',val_label.count(1))
    print('val Normal:',val_label.count(2))
    test_label = [label_dict[case] for case in test_path]
    print('test CP:',test_label.count(0))
    print('test NCP:',test_label.count(1))
    print('test Normal:',test_label.count(2))


    train_cp = [case for case in train_path if label_dict[case] == 0]
    train_cp = [os.path.basename(case).split('_')[0]for case in train_cp]
    train_cp = len(set(train_cp))
    train_ncp = [case for case in train_path if label_dict[case] == 1]
    train_ncp = [os.path.basename(case).split('_')[0]for case in train_ncp]
    train_ncp = len(set(train_ncp))
    train_normal = [case for case in train_path if label_dict[case] == 2]
    train_normal = [os.path.basename(case).split('_')[0]for case in train_normal]
    train_normal = len(set(train_normal))

    print('Train CP patient number:%d'%train_cp)
    print('Train NCP patient number:%d'%train_ncp)
    print('Train Normal patient number:%d'%train_normal)

    val_cp = [case for case in validation_path if label_dict[case] == 0]
    val_cp = [os.path.basename(case).split('_')[0]for case in val_cp]
    val_cp = len(set(val_cp))
    val_ncp = [case for case in validation_path if label_dict[case] == 1]
    val_ncp = [os.path.basename(case).split('_')[0]for case in val_ncp]
    val_ncp = len(set(val_ncp))
    val_normal = [case for case in validation_path if label_dict[case] == 2]
    val_normal = [os.path.basename(case).split('_')[0]for case in val_normal]
    val_normal = len(set(val_normal))

    print('Val CP patient number:%d'%val_cp)
    print('Val NCP patient number:%d'%val_ncp)
    print('Val Normal patient number:%d'%val_normal)

    test_cp = [case for case in test_path if label_dict[case] == 0]
    test_cp = [os.path.basename(case).split('_')[0]for case in test_cp]
    test_cp = len(set(test_cp))
    test_ncp = [case for case in test_path if label_dict[case] == 1]
    test_ncp = [os.path.basename(case).split('_')[0]for case in test_ncp]
    test_ncp = len(set(test_ncp))
    test_normal = [case for case in test_path if label_dict[case] == 2]
    test_normal = [os.path.basename(case).split('_')[0]for case in test_normal]
    test_normal = len(set(test_normal))

    print('Test CP patient number:%d'%test_cp)
    print('Test NCP patient number:%d'%test_ncp)
    print('Test Normal patient number:%d'%test_normal)

    return train_path, validation_path,test_path


if __name__ == "__main__":
  
  file_path = '/staff/shijun/torch_projects/COVID-19_CLS/converter/csv_file/new_resize_shuffle_label.csv'
  label_dict = csv_reader_single(file_path,key_col='id',value_col='label')

  path_list = list(label_dict.keys())
  train,val,_ = get_cross_validation_on_patient(path_list,6,2,label_dict=label_dict)
  
