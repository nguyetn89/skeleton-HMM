''' Skeleton-based Abnormal Gait Detection (Sensors, MDPI 2016)
    BSD 2-Clause "Simplified" License
    Author: Trong-Nguyen Nguyen'''

from __future__ import division
import numpy as np
from utils import *
from func_HMM import *
import argparse, sys

def HMMgait3_fullsequence(skel_dataset, test_subjects, n_subjects = 9, n_gaits = 9, window = 5, state_num = 24, obs_num = 43, save_result = False):
    if isinstance(test_subjects, int):
        test_subjects = [test_subjects]
    print('test subject(s): ' + str(test_subjects))
    '''define variables'''
    train_vectors = np.array([]).reshape((0,8))
    train_files = np.array([]).reshape((1,0))
    file_id = 0
    '''load data for training'''
    print('Load normal gaits of ' + str(n_subjects - len(test_subjects)) + ' subjects for training...')
    for i in range(n_subjects):
        if i in test_subjects:
            continue
        file_id += 1
        print('processing normal skel. of subject ' + str(i))
        tmp_skels = skel_dataset[i, 0]
        tmp = np.array([feature_extract(skel) for skel in tmp_skels])
        train_vectors = np.concatenate((train_vectors, tmp), axis = 0)
        train_files = np.concatenate((train_files, file_id*np.ones((1, tmp.shape[0]))), axis = 1)
    '''load test data'''
    print('Load test data...')
    abnormal_test_vectors = []
    normal_test_vectors = []
    n_abnormal_test_sequence = len(test_subjects) * (n_gaits - 1)
    n_normal_test_sequence = len(test_subjects)
    for i in range(n_subjects):
        if i in test_subjects:
            print('processing skel. of subject ' + str(i))
            for j in range(n_gaits):
                tmp_skels = skel_dataset[i, j]
                tmp = np.array([feature_extract(skel) for skel in tmp_skels])
                if j != 0:
                    abnormal_test_vectors.append(tmp)
                else:
                    normal_test_vectors.append(tmp)
    '''k-means'''
    print('window width = ' + str(window) + ', states = ' + str(state_num) + ', observations = ' + str(obs_num))
    centroids, train_labels, kmeans = KMeans_postures(train_vectors, obs_num)
    '''Training cycles'''
    train_cycle_vectors = []
    cycles_id = get_cycles(train_vectors, train_files, window)
    cycles_idx = np.where(cycles_id != -1)[0]
    for i in range(len(cycles_idx)-1):
        if cycles_id[cycles_idx[i]] == cycles_id[cycles_idx[i+1]]:
            train_cycle_vectors.append(train_labels[cycles_idx[i]:cycles_idx[i+1]+1])
    '''Train HMM'''
    hmm_model = train_hmm(train_cycle_vectors, state_num, obs_num)
    '''Testing abnormal cycles'''
    scores_abnormal = np.array([])
    groundtruth_cycle_abnormal = np.array([])
    scores_cycle_abnormal = np.array([])
    for f in range(n_abnormal_test_sequence):
        tmp_result = np.array([])
        test_labels = KMeans_test(kmeans, np.concatenate(abnormal_test_vectors, axis = 0))
        cycles_id = get_cycles(abnormal_test_vectors[f], f*np.ones(abnormal_test_vectors[f].shape[0]), window)
        cycles_idx = np.where(cycles_id != -1)[0]
        for i in range(len(cycles_idx)-1):
            if cycles_id[cycles_idx[i]] == cycles_id[cycles_idx[i+1]]:
                tmp_vt = test_labels[cycles_idx[i]:cycles_idx[i+1]+1]
                ll_val = hmm_model.score(tmp_vt.reshape((1,-1)))
                tmp_result = np.append(tmp_result, ll_val)
        scores_abnormal = np.append(scores_abnormal, -seq_likelihood(tmp_result,3))

        tmp_result = tmp_result[abs(tmp_result)!=np.inf]
        groundtruth_cycle_abnormal = np.concatenate((groundtruth_cycle_abnormal, np.ones(len(tmp_result))), axis = 0)
        scores_cycle_abnormal = np.concatenate((scores_cycle_abnormal, -np.array(tmp_result)), axis = 0)
    '''Testing normal cycles'''
    scores_normal = np.array([])
    groundtruth_cycle_normal = np.array([])
    scores_cycle_normal = np.array([])
    for f in range(n_normal_test_sequence):
        tmp_result = np.array([])
        test_labels = KMeans_test(kmeans, np.concatenate(normal_test_vectors, axis = 0))
        cycles_id = get_cycles(normal_test_vectors[f], f*np.ones(normal_test_vectors[f].shape[0]), window)
        cycles_idx = np.where(cycles_id != -1)[0]
        for i in range(len(cycles_idx)-1):
            if cycles_id[cycles_idx[i]] == cycles_id[cycles_idx[i+1]]:
                tmp_vt = test_labels[cycles_idx[i]:cycles_idx[i+1]+1]
                ll_val = hmm_model.score(tmp_vt.reshape((1,-1)))
                tmp_result = np.append(tmp_result, ll_val)
        scores_normal = np.append(scores_normal, -seq_likelihood(tmp_result,3))

        tmp_result = tmp_result[abs(tmp_result)!=np.inf]
        groundtruth_cycle_normal = np.concatenate((groundtruth_cycle_normal, -np.ones(len(tmp_result))), axis = 0)
        scores_cycle_normal = np.concatenate((scores_cycle_normal, -np.array(tmp_result)), axis = 0)
    '''ASSESSMENT ON TEST SET'''
    print('TEST RESULTS')
    auc, eer, eer_expected, sensitivity, specificity, precision, accuracy, F1 = assessment(scores_abnormal, scores_normal)
    result_seq = [auc, eer, eer_expected, sensitivity, specificity, precision, accuracy, F1]
    print('Full sequence:   AUC = %.3f --- EER = %.3f' % (auc,eer))
    auc, eer, eer_expected, sensitivity, specificity, precision, accuracy, F1 = assessment(scores_cycle_abnormal, scores_cycle_normal)
    result_cycle = [auc, eer, eer_expected, sensitivity, specificity, precision, accuracy, F1]
    print('Cycle:           AUC = %.3f --- EER = %.3f' % (auc,eer))
    '''write to file'''
    if save_result:
        filename = 'test_subject_' + ''.join(map(str, test_subjects)) + '.txt'
        write_results_to_file(filename, np.concatenate(([window, state_num, obs_num], result_seq, result_cycle)))

def main(argv):
    np.random.seed(1993)
    '''usage: python main.py -l 0 -w 5 -s 24 -o 43 -f 0'''
    parser = argparse.ArgumentParser(description = 'skeleton-based abnormal gait detection')
    parser.add_argument('-l', '--l1o', help = 'perform leave-one-out cross-validation', required = True)
    parser.add_argument('-w', '--width', help = 'window width for smoothing', required = True)
    parser.add_argument('-s', '--states', help = 'number of HMM states', required = True)
    parser.add_argument('-o', '--observations', help = 'number of HMM observations', required = True)
    parser.add_argument('-f', '--file', help = 'save result to file', required = True)
    args = vars(parser.parse_args())
    '''read and assign arguments'''
    l1o = bool(int(args['l1o']))
    window = int(args['width'])
    state_num = int(args['states'])
    obs_num = int(args['observations'])
    save_result = bool(int(args['file']))
    '''main processing'''
    data_path = 'dataset/DIRO_skeletons.npz'
    loaded = np.load(data_path)
    data, separation = loaded['data'], loaded['split']
    if l1o:
        for i in range(data.shape[0]):
            HMMgait3_fullsequence(data, i, n_subjects = data.shape[0], n_gaits = data.shape[1],\
                window = window, state_num = state_num, obs_num = obs_num, save_result = save_result)
    else:
        HMMgait3_fullsequence(data, np.where(separation == 'test')[0], n_subjects = data.shape[0], n_gaits = data.shape[1],\
            window = window, state_num = state_num, obs_num = obs_num, save_result = save_result)

if __name__ == '__main__':
    main(sys.argv)
