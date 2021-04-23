import h5py
import numpy as np
from scipy.spatial.distance import cdist
import json
import re
import random
import math
import pickle

def save_json(data, file_path):
    with open(file_path, "w") as f:
        json.dump(data, f)

def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def cosine_similarity(x, y):
    return 1. - cdist(np.array(x).reshape(1,768), np.array(y).reshape(1,768), 'cosine')

def calc_cross_sim(A, B):
    sum = 0.0
    index_A = random.sample(range(0, len(A)), min(100,len(A)))
    index_B = random.sample(range(0, len(B)), min(100,len(B)))
    for i in range(min(100,len(A))):
        for j in range(min(100,len(B))):
            sum = sum + cosine_similarity(A[index_A[i]],B[index_B[j]])

    return sum / (min(100,len(A))*min(100,len(B)))

def calc_self_sim(A):
    sum = 0.0
    if len(A)==1:
        return 0
    index_A = random.sample(range(0, len(A)), min(100,len(A)))
    for i in range(min(100,len(A))):
        for j in range(min(100,len(A))):
            if i!=j:
                sum = sum + cosine_similarity(A[index_A[i]],A[index_A[j]])

    return sum / ((min(100,len(A))-1)*(min(100,len(A))))
def find_pos_cur(a, i):
    cnt = -1
    while True:
        if i==-1 or a[i]=='<eos>':
            break
        i = i - 1
        cnt = cnt + 1
    return cnt

def find_pos_prev(a, i):
    cnt = -1
    while True:
        if i==-1 or a[i]=='<eos>':
            break
        i = i - 1
        cnt = cnt + 1

    i = i - 1
    if i==-2:
        return -1

    while True:
        if i==-1 or a[i]=='<eos>':
            break
        i = i - 1
        cnt = cnt + 1
    return cnt
def make_sub_sim_dict(tvqa_plus_subtitle, file_name, file3, file4):
    
    sim_dict = {}
    cc = 0
    for k,v in tvqa_plus_subtitle.items():
        cc = cc + 1
        print(cc, len(tvqa_plus_subtitle.items()))
        subtitle = (v['sub_text'].split())
        subtitle = list(filter(('<eos>').__ne__, subtitle))
        for i in range(len(file4[k])):
            b = file4[k][i]
            sub = subtitle[i].lower()
            if sub in sim_dict:
                sim_dict[sub].append(b)
            else:
                sim_dict[sub] = [b]
    with open(file_name,'wb') as fw:
        pickle.dump(sim_dict, fw)
    return sim_dict

def make_qa_sim_dict(tvqa_plus_qa, file_name, file3, file4):
        
    sim_dict = {}
    cc = 0
    for qa_dict in tvqa_plus_qa:
        cc = cc + 1
        print(cc,len(tvqa_plus_qa))
        qa = []
        qa.append(qa_dict['s_tokenized_q'].split())
        qa.append(qa_dict['s_tokenized_a0'].split())
        qa.append(qa_dict['s_tokenized_a1'].split())
        qa.append(qa_dict['s_tokenized_a2'].split())
        qa.append(qa_dict['s_tokenized_a3'].split())
        qa.append(qa_dict['s_tokenized_a4'].split())
        qid = qa_dict['q_id']
        keys = []
        keys.append(str(qid) + '_q')
        keys.append(str(qid) + '_a0')
        keys.append(str(qid) + '_a1')
        keys.append(str(qid) + '_a2')
        keys.append(str(qid) + '_a3')
        keys.append(str(qid) + '_a4')

        for (n,k) in enumerate(keys):
            for i in range(len(file3[k])):
                b = file3[k][i]
                qa_token = qa[n][i].lower()
                if qa_token in sim_dict:
                    sim_dict[qa_token].append(b)
                else:
                    sim_dict[qa_token] = [b]

    with open(file_name,'wb') as fw:
        pickle.dump(sim_dict, fw)
    return sim_dict

def check_name_sim(sim_dict, file3, file4):
    print(calc_cross_sim(sim_dict['at'], sim_dict['if']))
    name_list_lower2 = ['at', ':', 'you', 'me', 'i'] + name_list_lower
    print(name_list_lower2)
    for name in name_list_lower2:
        sum = 0.0
        for name2 in name_list_lower:
            if name!=name2 and (name in sim_dict) and (name2 in sim_dict):
                sum = sum + calc_cross_sim(sim_dict[name], sim_dict[name2])
        
        if (name in sim_dict):
            print(name, calc_self_sim(sim_dict[name]), sum / (len(name_list_lower)-1)) 

def check_youI(tvqa_plus_subtitle, file3, file4):
    
    sum_me = 0
    cnt_me = 0
    sum_i = 0
    cnt_i = 0
    sum_you = 0
    cnt_you = 0
    prev_k = ''
    prev_subtitle = []
    cc = 0
    for k,v in tvqa_plus_subtitle.items():
        cc = cc + 1
    #    print(cc, len(tvqa_plus_subtitle.items()))
        subtitle = (v['sub_text'].split())
        subtitle_eos = subtitle
        subtitle = list(filter(('<eos>').__ne__, subtitle))
        i_eos = 0
        for i in range(len(file4[k])):
            sub = subtitle[i].lower()
            if subtitle_eos[i_eos] == '<eos>':
                i_eos = i_eos + 1
            if sub=='me':
                sum_me = sum_me + cosine_similarity(file4[k][i],file4[k][i-find_pos_cur(subtitle_eos, i_eos)])
                cnt_me = cnt_me + 1
                #print(subtitle[i], subtitle[i-find_pos_cur(subtitle_eos, i_eos)])
            if sub=='i':
                sum_i = sum_i + cosine_similarity(file4[k][i],file4[k][i-find_pos_cur(subtitle_eos, i_eos)])
                cnt_i = cnt_i + 1
                #print(subtitle[i], subtitle[i-find_pos_cur(subtitle_eos, i_eos)])
            if sub=='you' and find_pos_prev(subtitle_eos, i_eos)!=-1:
                sum_you = sum_you + cosine_similarity(file4[k][i],file4[k][i-find_pos_prev(subtitle_eos, i_eos)])
                cnt_you = cnt_you + 1

                #print(subtitle[i], subtitle[i-find_pos_prev(subtitle_eos, i_eos)])

            i_eos = i_eos + 1
        prev_subtitle = subtitle
        #if cc == 100:
        #    break
    print(cnt_me, cnt_i, cnt_you, sum_me/cnt_me, sum_i/cnt_i, sum_you/cnt_you)


file3 = h5py.File('../../../../home_klimt/dohyun.kim/tvqa_plus_stage_features/qa14_1_200_8_True.h5', "r")  # qid + key
file4 = h5py.File('../../../../home_klimt/dohyun.kim/tvqa_plus_stage_features/sub14_1_200_8_True.h5', "r")  # qid + key

#file3 = h5py.File('../feature_extract/tvqa_plus_stage_features/qa.h5', "r")  # qid + key
#file4 = h5py.File('../feature_extract/tvqa_plus_stage_features/sub.h5', "r")  # qid + key
tvqa_plus_subtitle = load_json('../feature_extract/tvqa_plus_stage_features/tvqa_plus_subtitles.json')
tvqa_plus_qa = load_json('../feature_extract/qa.json')

#a = file4['s09e14_seg02_clip_04'][137] #Sheldon
name_list = (load_json('../feature_extract/./name.json')).keys()
name_list_lower = [x.lower() for x in name_list]
sim_dict = make_sub_sim_dict(tvqa_plus_subtitle, 'sub_loss14.pickle', file3, file4)
#sim_dict = make_qa_sim_dict(tvqa_plus_qa, 'qa_sim_youI.pickle, file3, file4)

with open('sub_loss14.pickle','rb') as fr:
    sim_dict = pickle.load(fr)

for (k,v) in tvqa_plus_subtitle.items():
    print(cosine_similarity(file4[k][48], file4[k][30]))
    print(cosine_similarity(file4[k][0], file4[k][30]))
    break
check_name_sim(sim_dict, file3, file4)
check_youI(tvqa_plus_subtitle, file3, file4)
