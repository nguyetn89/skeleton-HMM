from __future__ import division
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import scipy.interpolate
import matplotlib.pyplot as plt
from ROC import *

'''determine {a, b, c, d} of plane ax+by+cz+d=0'''
def eq_plane(p1, p2, p3):
    v1 = p3 - p1
    v2 = p2 - p1
    # normal vector
    nv = np.cross(v1, v2)
    a, b, c = nv
    # calc d = -(ax + by + cz)
    d = -np.dot(nv, p3)
    return [a, b, c, d]

def calc_angle_between_planes(p1A, p2A, p3A, p1B, p2B, p3B):
    p1 = eq_plane(p1A, p2A, p3A)
    p2 = eq_plane(p1B, p2B, p3B)
    v1 = np.array(p1[:3])
    v2 = np.array(p2[:3])
    mult = np.dot(v1, v2)
    len_v1 = np.sum(v1**2)**0.5
    len_v2 = np.sum(v2**2)**0.5
    return np.arccos(mult / (len_v1 * len_v2))

def calc_angle_3_points(p1, p2, p3, angle_of_vector = True):
    v1 = p1 - p2
    v2 = p3 - p2
    mult = np.dot(v1, v2)
    if not angle_of_vector:
        mult = abs(mult)
    len_v1 = np.sum(v1**2)**0.5
    len_v2 = np.sum(v2**2)**0.5
    return np.arccos(mult / (len_v1 * len_v2))

#work with Kinect 2 joint's indices (see Microsoft website)
def feature_extract(skel):
    '''foot'''
    id = 15
    foot_left = skel[id*3:id*3+3]
    id = 19
    foot_right = skel[id*3:id*3+3]
    '''ankle'''
    id = 14
    ankle_left = skel[id*3:id*3+3]
    id = 18
    ankle_right = skel[id*3:id*3+3]
    '''knee'''
    id = 13
    knee_left = skel[id*3:id*3+3]
    id = 17
    knee_right = skel[id*3:id*3+3]
    '''hip'''
    id = 12
    hip_left = skel[id*3:id*3+3]
    id = 16
    hip_right = skel[id*3:id*3+3]
    '''torso'''
    id = 0
    hip_center = skel[id*3:id*3+3]
    '''feature extraction'''
    vt = np.zeros(8)
    vt[0] = calc_angle_3_points(foot_left, ankle_left, knee_left)
    vt[1] = calc_angle_3_points(foot_right, ankle_right, knee_right)
    vt[2] = calc_angle_3_points(ankle_left, knee_left, hip_left)
    vt[3] = calc_angle_3_points(ankle_right, knee_right, hip_right)
    vt[4] = calc_angle_3_points(knee_left, hip_left, hip_center)
    vt[5] = calc_angle_3_points(knee_right, hip_right, hip_center)
    vt[6] = calc_angle_between_planes(ankle_left, knee_left, hip_left, ankle_right, knee_right, hip_right)
    '''dist(two ankles)'''
    vt[7] = np.sum((ankle_left - ankle_right)**2)**0.5
    return vt.reshape(-1)

'''this func is useful if vectors were combined from different sequences'''
def get_cycles(vectors, id_files, w, show_vectors_shape = False):
    if show_vectors_shape:
        print('vectors.shape = ' + str(vectors.shape))
    seq = np.reshape(vectors[:,-1],-1)
    seq = smooth_seq(seq)
    cycles = -np.ones(seq.size)
    id_files = id_files.reshape(-1)
    for i in range(w//2,len(seq)-w//2+1):
        if seq[i] > np.amax(np.concatenate((seq[i-w//2:i], seq[i+1:i+w//2+1]), axis = 0)) and len(np.unique(id_files[i-w//2:i+w//2+1])) == 1:
            cycles[i] = id_files[i]
    return cycles

def smooth_seq(input_seq, seg_len = 1200, w = 5):
    assert w % 2 == 1
    output_seq = np.zeros(input_seq.size)
    for s in range(input_seq.size//seg_len):
        tmp_in = np.copy(input_seq[s*seg_len:(s+1)*seg_len])
        tmp_out = np.zeros(tmp_in.size)
        for i in range(tmp_out.size):
            if i < w//2:
                tmp_out[i] = np.mean(tmp_in[:i*2+1])
            elif i + w//2 > tmp_out.size - 1:
                tmp_w = tmp_out.size - 1 - i
                tmp_out[i] = np.mean(tmp_in[i - tmp_w:])
            else:
                tmp_out[i] = np.mean(tmp_in[i-w//2:i+w//2+1])
        output_seq[s*seg_len:(s+1)*seg_len] = list(tmp_out)
    return output_seq

def assessment(prob_list_abnormal, prob_list_normal):
    prob_list_abnormal[np.isnan(prob_list_abnormal)] = np.inf
    max_val = np.amax(prob_list_abnormal[prob_list_abnormal!=np.inf])
    prob_list_abnormal[prob_list_abnormal==np.inf] = max_val

    prob_list_normal[np.isnan(prob_list_normal)] = np.inf
    max_val = np.amax(prob_list_normal[prob_list_normal!=np.inf])
    prob_list_normal[prob_list_normal==np.inf] = max_val
    #
    return assessment_unit(prob_list_abnormal, prob_list_normal)
    
def write_results_to_file(filename, data):
    with open(filename, "a") as myfile:
        for i in range(len(data)):
            myfile.write(str(data[i]))
            if i < len(data) - 1:
                myfile.write(',')
        myfile.write('\n')
