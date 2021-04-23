1, prepare data\
https://drive.google.com/file/d/1GnknXfs9qKE-WVaUgUeKfCTLHjyzqCHG/view 에서 파일 다운로드 후 압축을 푼다.
```
tar -xf tvqa_plus_stage_features_new.tar.gz
```
이후 tvqa_plus_stage_features 폴더를 ./feature_extract에 위치시킨다.

2, prepare qa.json, subtitle.json
```
python make_json.py
```

3, prepare pretraining text
```
python generate_pretraining_text.py
```
4, fine-tuning with pretraining text
```
srun1 --qos=ilow python run_lm_finetuning.py \
  --bert_model bert-base-uncased \
  --do_train \
  --train_file pretraining_text.txt \
  --output_dir ./result \
  --num_train_epochs 3.0 \
  --learning_rate 3e-5 \
  --train_batch_size 32 \
  --max_seq_length 128 \
--do_lower_case  \
--seed 44 
```
5, extract 768D vector from qa, subtitle (reproduced version)
```
srun1 --qos=ilow python extract_tokenized_tvqa_features.py --mode qa --output_file ./tvqa_plus_stage_features/qa.h5 --bert_model ../feature_extract/result/
srun1 --qos=ilow python extract_tokenized_tvqa_features.py --mode sub --output_file ./tvqa_plus_stage_features/sub.h5 --bert_model ../feature_extract/result/
```

---------------------------------------------------------------------------------------------
실험때마다 이걸 반복하면 됩니다.\
5, extract 768D vector from qa, subtitle (character bert version)
```
srun1 --qos=ilow python extract_tokenized_tvqa_features_unk.py --bert_model ../feature_extract/result/ --train --inference --ratio 1 --epoch 200 --batch_size 8 --use_youI --save_path ../../../../home_klimt/dohyun.kim/tvqa_plus_stage_features/
```











source ~/.bashrc ; conda activate py_36_2 ; cd stage_reproduced/feature_extract ; srun1 --qos=ilow python extract_tokenized_tvqa_features_unk.py --bert_model ../feature_extract/result/ --train --inference --ratio 1 --epoch 200 --batch_size 8 --use_youI

source ~/.bashrc ; conda activate tvqa_plus ; cd stage_reproduced/stage ; bash run_main.sh --add_local