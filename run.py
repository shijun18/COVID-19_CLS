import os
import numpy as np
import argparse
from trainer import VolumeClassifier
import pandas as pd
from data_utils.csv_reader import csv_reader_single
from config import INIT_TRAINER, SETUP_TRAINER, VERSION, CURRENT_FOLD

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import time
import random
from utils import exclude_path



def get_cross_validation_on_patient(path_list, fold_num, current_fold, label_dict):

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
    print("Train set length: ", len(train_path),
          "\nVal set length:", len(validation_path),
          '\nTest set len:',len(test_path))
    
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

    return train_path, validation_path


def get_cross_validation(path_list, fold_num, current_fold):

    _len_ = len(path_list) // fold_num
    train_id = []
    validation_id = []
    end_index = current_fold * _len_
    start_index = end_index - _len_
    if current_fold == fold_num:
        validation_id.extend(path_list[start_index:])
        train_id.extend(path_list[:start_index])
    else:
        validation_id.extend(path_list[start_index:end_index])
        train_id.extend(path_list[:start_index])
        train_id.extend(path_list[end_index:])

    print(len(train_id), len(validation_id))
    return train_id, validation_id


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-m',
                        '--mode',
                        default='train',
                        choices=["train-cross","train", "inf"],
                        help='choose the mode',
                        type=str)
    parser.add_argument('-s',
                        '--save',
                        default='no',
                        choices=['no', 'n', 'yes', 'y'],
                        help='save the forward middle features or not',
                        type=str)
    args = parser.parse_args()

    # Set data path & classifier
    ###### modification for new data
    old_csv_path = './converter/shuffle_label.csv'
    label_dict = csv_reader_single(old_csv_path, key_col='id', value_col='label')

    # new_csv_path = './converter/shuffle_label.csv'
    # new_csv_path = './converter/new_shuffle_label.csv'
    new_csv_path = './converter/new_resize_shuffle_label.csv'
    total_label_dict = csv_reader_single(new_csv_path, key_col='id', value_col='label')

    ######

    classifier = VolumeClassifier(**INIT_TRAINER)
    print(get_parameter_number(classifier.net))

    # Training
    ###############################################
    if args.mode == 'train-cross':
        path_list = list(total_label_dict.keys())
        for i in range(5):
            print('===================fold %d==================='%(i+1))
            train_path, val_path = get_cross_validation_on_patient(path_list, 6, i+1, total_label_dict)
            # train_path, val_path = get_cross_validation(path_list, 6, i+1)
            SETUP_TRAINER['train_path'] = train_path
            SETUP_TRAINER['val_path'] = val_path
            SETUP_TRAINER['label_dict'] = total_label_dict
            SETUP_TRAINER['cur_fold'] = i

            start_time = time.time()
            classifier.trainer(**SETUP_TRAINER)

            print('run time:%.4f' % (time.time() - start_time))
    
    elif args.mode == 'train':
        path_list = list(total_label_dict.keys())
        train_path, val_path = get_cross_validation_on_patient(path_list, 6, CURRENT_FOLD+1,total_label_dict)
        SETUP_TRAINER['train_path'] = train_path
        SETUP_TRAINER['val_path'] = val_path
        SETUP_TRAINER['label_dict'] = total_label_dict
        SETUP_TRAINER['cur_fold'] = CURRENT_FOLD

        start_time = time.time()
        classifier.trainer(**SETUP_TRAINER)

        print('run time:%.4f' % (time.time() - start_time))
    ###############################################

    # Inference
    ###############################################
    elif args.mode == 'inf':
        ex_path = exclude_path(old_csv_path,new_csv_path,'id')
        test_path = list(label_dict.keys())[3600:] + ex_path
        print('test len:',len(test_path))
        save_path = './analysis/new_result/{}.csv'.format(VERSION)

        start_time = time.time()
        if args.save == 'no' or args.save == 'n':
            result, _, _ = classifier.inference(test_path, total_label_dict)
            print('run time:%.4f' % (time.time() - start_time))
        else:
            result, feature_in, feature_out = classifier.inference(
                test_path, total_label_dict, hook_fn_forward=True)
            print('run time:%.4f' % (time.time() - start_time))
            # save the avgpool output
            print(feature_in.shape, feature_out.shape)
            feature_dir = './analysis/mid_feature/{}'.format(VERSION)
            if not os.path.exists(feature_dir):
                os.makedirs(feature_dir)
            from converter.common_utils import save_as_hdf5
            for i in range(len(test_path)):
                name = os.path.basename(test_path[i])
                feature_path = os.path.join(feature_dir, name)
                save_as_hdf5(feature_in[i], feature_path, 'feature_in')
                save_as_hdf5(feature_out[i], feature_path, 'feature_out')
        result['path'] = test_path
        csv_file = pd.DataFrame(result)
        csv_file.to_csv(save_path, index=False)
        #report
        cls_report = classification_report(
            result['true'],
            result['pred'],
            target_names=['CP', 'NCP', 'Normal'],
            output_dict=True)
        print(cls_report)
        cm = confusion_matrix(result['true'], result['pred'])
        print(cm)
        cls_report['CP']['specificity'] = np.sum(cm[1:,1:])/np.sum(cm[1:])
        cls_report['NCP']['specificity'] = (cm[0,0] + cm[0,2] + cm[2,0] + cm[2,2])/(np.sum(cm[0]) + np.sum(cm[2]))
        cls_report['Normal']['specificity'] = np.sum(cm[0:2,0:2])/np.sum(cm[:2])
        #save as csv
        report_save_path = './analysis/new_result/{}_report.csv'.format(VERSION)
        report_csv_file = pd.DataFrame(cls_report)
        report_csv_file.to_csv(report_save_path)
    ###############################################
