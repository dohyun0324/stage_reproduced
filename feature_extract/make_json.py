import json
import re
def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def save_json(data, file_path):
    with open(file_path, "w") as f:
        json.dump(data, f)

tvqa_plus_train = load_json('./tvqa_plus_train_preprocessed.json')
tvqa_plus_valid = load_json('./tvqa_plus_valid_preprocessed.json')
tvqa_plus_test = load_json('./tvqa_plus_test_preprocessed_no_anno.json')
tvqa_plus_subtitle = load_json('./tvqa_plus_subtitles.json')

tvqa = tvqa_plus_train + tvqa_plus_valid + tvqa_plus_test
#print(tvqa_plus_subtitle['s10e13_seg02_clip_14']['sub_time'])
qa = []
for i in range(len(tvqa)):
    qa.append({'q_id' : tvqa[i]['qid'], 's_tokenized_q' : tvqa[i]['q'], 's_tokenized_a0' : tvqa[i]['a0'], 's_tokenized_a1' : tvqa[i]['a1'], 's_tokenized_a2' : tvqa[i]['a2'], 's_tokenized_a3' : tvqa[i]['a3'], 's_tokenized_a4' : tvqa[i]['a4']})

subtitle = []
for (k,v) in sorted(tvqa_plus_subtitle.items()):
    subtitle.append({'vid_name' : k, 's_tokenized_sub_text' : v["sub_text"]})

save_json(subtitle, './subtitle.json')
save_json(qa, './qa.json')