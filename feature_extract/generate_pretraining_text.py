import json
import re
def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

tvqa_plus_train = load_json('tvqa_plus_train_preprocessed.json')
tvqa_plus_subtitle = load_json('tvqa_plus_subtitles.json')
#print(tvqa_plus_train[0]['answer_idx'])
#print(len(tvqa_plus_subtitle))
f = open("pretraining_text.txt","w")
map = {}
cnt = 0
for i in range(23545):
    subtitle_name = tvqa_plus_train[i]['vid_name']
    f.write(tvqa_plus_train[i]['q'] + '\n' + tvqa_plus_train[i]['a'+str(tvqa_plus_train[i]['answer_idx'])] + '\n')
    f.write('\n')

for (k,v) in tvqa_plus_subtitle.items():
#    if not (k in map):
    f.write(v["sub_text"].replace(' <eos> ','\n') + '\n')
    f.write('\n')