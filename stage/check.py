
import h5py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
import re
def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)
#file3 = h5py.File('../../../../home_klimt/dohyun.kim/tvqa_plus_stage_features/qa_processed.h5', "r")  # qid + key
#file4 = h5py.File('../../../../home_klimt/dohyun.kim/tvqa_plus_stage_features/sub_processed.h5', "r")  # qid + key

file3 = h5py.File('../feature_extract/tvqa_plus_stage_features/qa.h5', "r")  # qid + key
file4 = h5py.File('../feature_extract/tvqa_plus_stage_features/sub.h5', "r")  # qid + key
tvqa_plus_subtitle = load_json('../feature_extract/tvqa_plus_stage_features/tvqa_plus_subtitles.json')

a = file4['s09e14_seg02_clip_04'][137] #Sheldon
sum_unk = 0
cnt_unk = 0
sum_sheldon = 0
cnt_sheldon = 0
cc = 0
for k,v in tvqa_plus_subtitle.items():
    cc = cc + 1
    #print(cc, len(tvqa_plus_subtitle.items()))
    subtitle = (v['sub_text'].split())
    subtitle = list(filter(('<eos>').__ne__, subtitle))
    for i in range(len(file4[k])):
        b = file4[k][i]
        if subtitle[i]=='UNKNAME':
            sum_unk = sum_unk + cosine_similarity([a],[b])
            cnt_unk = cnt_unk + 1
        if subtitle[i]=='Sheldon':
            sum_sheldon = sum_sheldon + cosine_similarity([a],[b])
            cnt_sheldon = cnt_sheldon + 1

    print(sum_unk / cnt_unk, sum_sheldon / cnt_sheldon)
#print(np.random.normal(-0.03,0.7, size = (15,768)).tolist())
