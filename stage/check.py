import h5py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
import re
def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)
'''
file1 = h5py.File('../../../../home_klimt/dohyun.kim/tvqa_plus_stage_features/bbt_qa_s_tokenized_bert_sub_qa_tuned_new_qid.h5', "r")  # qid + key
file2 = h5py.File('../../../../home_klimt/dohyun.kim/tvqa_plus_stage_features/bbt_sub_s_tokenized_bert_sub_qa_tuned.h5', "r")  # qid + key
file3 = h5py.File('../../../../home_klimt/dohyun.kim/tvqa_plus_stage_features/bbt_qa_s_tokenized_bert_sub_qa_tuned_new_qid_random_01.h5', "w")  # qid + key
file4 = h5py.File('../../../../home_klimt/dohyun.kim/tvqa_plus_stage_features/bbt_sub_s_tokenized_bert_sub_qa_tuned_random_01.h5', "w")  # qid + key
for key in file1.keys():
    shape = file1[key].shape
    data = np.random.normal(0,1, size = shape).tolist()
    file3.create_dataset(key,data = data, dtype = 'f4')

for key in file2.keys():
    shape = file2[key].shape
    data = np.random.normal(0,1, size = shape).tolist()
    file4.create_dataset(key,data = data, dtype = 'f4')
'''
#print(file1['100017_a0'].shape) 
# print(file2.keys()[:10]) 


#file3 = h5py.File('../../../../home_klimt/dohyun.kim/tvqa_plus_stage_features/bbt_qa_s_tokenized_bert_sub_qa_tuned_new_qid_random.h5', "w")  # qid + key
#file4 = h5py.File('../../../../home_klimt/dohyun.kim/tvqa_plus_stage_features/bbt_sub_s_tokenized_bert_sub_qa_tuned_random.h5', "w")  # qid + key
#file3.create_dataset(file1, data = , dtype = 'f4')
#          qa_h5_file.create_dataset(str(qid) + "_q", data = embedding_q, dtype = 'f4')
file = h5py.File('../../../../home_klimt/dohyun.kim/tvqa_plus_stage_features/bbt_qa_s_tokenized_bert_sub_qa_tuned_new_qid.h5', "r")  # qid + key
file2 = h5py.File('../../../../home_klimt/dohyun.kim/tvqa_plus_stage_features/bbt_sub_s_tokenized_bert_sub_qa_tuned.h5', "r")  # qid + key
file3 = h5py.File('../../../../home_klimt/dohyun.kim/tvqa_plus_stage_features/qa_3.h5', "r")  # qid + key
file4 = h5py.File('../../../../home_klimt/dohyun.kim/tvqa_plus_stage_features/sub_3.h5', "r")  # qid + key
tvqa_plus_subtitle = load_json('../../../../home_klimt/dohyun.kim/tvqa_plus_stage_features/tvqa_plus_subtitles.json')
print(file['103392_q'][0][:10])
print(file3['103392_q'][0][:10])
print(file['103392_a0'])
print(file['103392_a1'])
print(file['103392_a2'])
print(np.mean(file['103392_a0']), np.std(file['103392_a0']))
print(np.mean(file2['s10e13_seg02_clip_10']), np.std(file2['s10e13_seg02_clip_10']))
print(np.mean(file3['103392_q']), np.std(file3['103392_q']), file3['103392_q'])
print(np.mean(file4['s10e13_seg02_clip_10']), np.std(file4['s10e13_seg02_clip_10']))
print(file2['s10e13_seg02_clip_14'].shape)
print(file4['s10e13_seg02_clip_14'].shape)
a = file4['s09e14_seg02_clip_00'][0] #Sheldon
for k,v in tvqa_plus_subtitle.items():
    subtitle = (v['sub_text'].split())
    subtitle = list(filter(('<eos>').__ne__, subtitle))
    for i in range(len(file4[k])):
#        if subtitle[i]=='UNKNAME' or subtitle[i]=='Sheldon:
        b = file4[k][i]
        print(k, i, cosine_similarity([a],[b]), subtitle[i])
#print(np.random.normal(-0.03,0.7, size = (15,768)).tolist())
