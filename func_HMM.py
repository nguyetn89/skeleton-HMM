from __future__ import division
import numpy as np
from hmmlearn import hmm, base
from sklearn.preprocessing import LabelEncoder
import sklearn.cluster


def seq_likelihood(seq, n = 3):
    if not isinstance(seq, np.ndarray):
        seq = np.array(seq)
    seq = seq[abs(seq)!=np.inf]
    if len(seq) < n:
        lh = np.mean(seq)
    else:
        tmp = np.zeros(len(seq)-n+1)
        for i in range(tmp.size):
            tmp[i] = np.mean(seq[i:i+n])
        lh = np.amin(tmp)
    return lh

def KMeans_test(kmeans, test_vts):
    if not isinstance(test_vts, np.ndarray):
        test_vts = np.array(test_vts)
    test_labels = kmeans.predict(test_vts[:,:-1])
    return test_labels

def KMeans_postures(train_vts, k):
    if not isinstance(train_vts, np.ndarray):
        train_vts = np.array(train_vts)
    kmeans = sklearn.cluster.KMeans(n_clusters = k).fit(train_vts[:,:-1])
    print('kmeans dimension: %d' % (kmeans.cluster_centers_.shape[1]))
    return kmeans.cluster_centers_, kmeans.labels_, kmeans

def train_hmm(input_data, state_num, obs_num):
    '''convert data to appropriate form'''
    data = np.array([]).reshape(-1)
    obs_lengths = []
    for vt in input_data:
        data = np.concatenate((data, vt.reshape(-1)), axis = 0)
        obs_lengths.append(len(vt))
    '''random weight initialization'''
    prior0 = np.zeros([1, state_num])
    prior0[0,0] = 1.
    transmat0 = np.random.rand(state_num, state_num)
    obsmat0 = np.random.rand(state_num, obs_num)
    '''specify structure'''
    for i in range(state_num):
        for j in range(state_num):
            if (j-i) not in [0, 1]:
                transmat0[i,j] = 0
    transmat0 = normalize_transmat(transmat0)
    obsmat0 = normalize_transmat(obsmat0)
    '''training'''
    model = hmm.MultinomialHMM(n_components = state_num, n_iter = 50)
    model.n_features = obs_num
    model.startprob_ = prior0
    model.transmat_ = transmat0
    model.emissionprob_ = obsmat0
    data = LabelEncoder().fit_transform(data)
    model.fit(np.atleast_2d(data).T, lengths = np.array(obs_lengths).reshape(-1))
    return model

'''sum each row is 1'''
def normalize_transmat(transmat):
    new_transmat = np.array([list(row/sum(row)) for row in transmat])
    return new_transmat
    
