import os
import pandas as pd 


csv_path = './unzip_filenames.csv'

csv_file = pd.read_csv(csv_path)

patient_id = csv_file['patient_id'].values.tolist()
scan_id = csv_file['scan_id'].values.tolist()

map_dict = {}

for scan, patient in zip(scan_id,patient_id):
    map_dict[str(scan)] = str(patient)


print(len(map_dict))
# print(map_dict)
'''
npy_path = '/staff/shijun/torch_projects/COVID-19_CLS/dataset/npy_data'

for case in os.scandir(npy_path):
    if os.path.isdir(case.path):
        for item in os.scandir(case.path):
            if '_' not in item.name:
                ID = item.name.split('.')[0]
                patient_id = map_dict[ID]
                new_path = os.path.join(case.path,'%s_%s'%(patient_id,item.name))
                print('old path:',item.path)
                print('new path:',new_path)
                os.rename(item.path, new_path)
                
'''

old_label_csv = './slice_number.csv'
new_label_csv = []
csv_file = pd.read_csv(old_label_csv)

col = list(csv_file.columns.values)

for _,line in csv_file.iterrows():
    item = []
    ID = os.path.basename(line['id']).split('.')[0]
    patient_id = map_dict[ID]
    item.append(line['id'].replace(ID,'%s_%s'%(patient_id,ID)))
    item.append(line['slice_num'])
    new_label_csv.append(item)
# print(new_label_csv)
new_file = pd.DataFrame(columns=col, data=new_label_csv)
new_file.to_csv(old_label_csv, index=False)