import json
import re
def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def save_json(data, file_path):
    with open(file_path, "w") as f:
        json.dump(data, f)

tvqa_plus_train = load_json('./tvqa_plus_stage_features/tvqa_plus_train_preprocessed.json')
tvqa_plus_subtitle = load_json('./tvqa_plus_stage_features/tvqa_plus_subtitles.json')
#print(tvqa_plus_train[0]['answer_idx'])
#print(len(tvqa_plus_subtitle))
f = open("pretraining_text_unk.txt","w")
map = {}
cnt = 0
for i in range(23545):
    subtitle_name = tvqa_plus_train[i]['vid_name']
    f.write(tvqa_plus_train[i]['q'] + '\n' + tvqa_plus_train[i]['a'+str(tvqa_plus_train[i]['answer_idx'])] + '\n')
    f.write('\n')

c = 0
name_list_lower = [x.lower() for x in load_json('./name.json').keys()]
for (k,v) in tvqa_plus_subtitle.items():
#    if not (k in map):
    sub = v["sub_text"]
    sub_split = sub.split()
    #print(len(name_index))
    for i in range(len(sub_split)):
        if sub_split[i].lower() in name_list_lower:
            sub_split[i] = 'UNKNAME'

    sub = ' '.join(sub_split)
    sub = sub.replace(' <eos> ', '\n')
    f.write(sub + '\n')
    f.write('\n')

#save_json(name_map, './name.json')
print(sub)