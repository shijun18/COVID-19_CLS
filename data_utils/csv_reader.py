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




if __name__ == "__main__":
  
  
  # Test sample - csv reader
  file_path = '/staff/shijun/torch_projects/COVID-19_CLS/converter/shuffle_label.csv'
  dict_ = csv_reader_single(file_path,key_col='id',value_col='label')
  print(list(dict_.keys())[0:10])
  
  # file_path = '/staff/shijun/torch_projects/COVID-19_CLS/converter/complete_info.csv'
  # dict_ = csv_reader_multi(file_path,key_col='patient_id',value_col=['Age','Sex'])
  # print(list(dict_.keys())[0:10])
  


