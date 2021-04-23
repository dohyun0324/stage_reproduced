from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
__author__ = "Jie Lei"
"""
Extract BERT contextualized token embedding, using [1]. Modified from [4].
Specially designed for extract TVQA text features.


Instructions:
0, This code should be running at Python 3.5+ and PyTorch 0.4.1/1.0
1, Input is a jsonl file. Each line is a json object, containing a 
   {"id": *str_id*, "text": *space separated sequence*}. Output is 
   also a jsonl file, each line contains 
2, Tokens that are split into subword by WordPiece, their embeddings are 
   the averaged embedding of its subword embeddings, as in the [2, 3]. This
   makes the output embedding respect the original tokenization scheme.
3, Each of the subtitle, question, and answers are encoded separately, 
   not in the form of sequence pairs. This means they are all `sequence A`, 
   `A embedding` is added to each of the token embeddings before forward 
   into the model. See [0] for details.
 

[0] BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
[1] https://github.com/huggingface/pytorch-pretrained-BERT
[2] From Recognition to Cognition: Visual Commonsense Reasoning
[3] SDNET: CONTEXTUALIZED ATTENTION-BASED DEEP NETWORK FOR CONVERSATIONAL QUESTION ANSWERING 
[4] https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples/extract_features.py
"""


from warmup_scheduler import GradualWarmupScheduler
import argparse
import logging
import warnings
import json
import numpy as np
from easydict import EasyDict as edict
from tqdm import tqdm
import base64
import h5py
import os
import sys
import torch.nn as nn
import torch.optim as optim
import time
from sklearn.metrics.pairwise import cosine_similarity
import math
import random
import sys

import torch
from torch.utils.data import Dataset, DataLoader, SequentialSampler

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
  
class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out
def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

name_map = load_json('./name.json')
def pad_sequences_1d(sequences, dtype=torch.long):
    """ Pad a single-nested list or a sequence of n-d torch tensor into a (n+1)-d tensor,
        only allow the first dim has variable lengths
    Args:
        sequences: list(n-d tensor or list)
        dtype: torch.long for word indices / torch.float (float32) for other cases
    Returns:
        padded_seqs: ((n+1)-d tensor) padded with zeros
        mask: (2d tensor) of the same shape as the first two dims of padded_seqs,
              1 indicate valid, 0 otherwise
    Examples:
        >>> test_data_list = [[1,2,3], [1,2], [3,4,7,9]]
        >>> pad_sequences_1d(test_data_list, dtype=torch.long)
        >>> test_data_3d = [torch.randn(2,3,4), torch.randn(4,3,4), torch.randn(1,3,4)]
        >>> pad_sequences_1d(test_data_3d, dtype=torch.float)
    """
    if isinstance(sequences[0], list):
        sequences = [torch.tensor(s, dtype=dtype) for s in sequences]
    extra_dims = sequences[0].shape[1:]  # the extra dims should be the same for all elements
    lengths = [len(seq) for seq in sequences]
    padded_seqs = torch.zeros((len(sequences), max(lengths)) + extra_dims, dtype=dtype)
    mask = torch.zeros(len(sequences), max(lengths)).float()
    for idx, seq in enumerate(sequences):
        end = lengths[idx]
        padded_seqs[idx, :end] = seq
        mask[idx, :end] = 1
    return padded_seqs, mask  # , lengths


def pad_collate(data):
    batch = edict()
    batch["token_ids"], batch["token_ids_mask"] = pad_sequences_1d([d.token_ids for d in data], dtype=torch.long)
    batch["token_map"] = [d.token_map for d in data]
    batch["unique_id"] = [d.unique_id for d in data]
    batch["tokens"] = [d.tokens for d in data]
    batch["original_tokens"] = [d.original_tokens for d in data]
    return batch


class BertSingleSeqDataset(Dataset):
    def __init__(self, input_file, bert_model, max_seq_len):
        if isinstance(input_file, str):
            self.examples = self.read_examples(input_file)
        elif isinstance(input_file, list):
            self.examples = input_file  # a list(dict) already
        else:
            raise TypeError("Expected str or list, got type {}".format(type(input_file)))
        print("There are {} lines".format(len(self.examples)))
        self.bert_full_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir=None)
        self.wordpiece_tokenizer = self.bert_full_tokenizer.wordpiece_tokenizer
        self.convert_tokens_to_ids = self.bert_full_tokenizer.convert_tokens_to_ids
        self.convert_ids_to_tokens = self.bert_full_tokenizer.convert_ids_to_tokens
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]
        original_tokens = example["text"].lower().split()
        wp_tokens, wp_token_ids, wp_map = self.normalize_text(original_tokens, self.max_seq_len)

        items = edict(
            unique_id=example["unique_id"],
            tokens=wp_tokens,
            token_ids=wp_token_ids,
            token_map=wp_map,
            original_tokens=original_tokens,
        )
        return items

    @staticmethod
    def read_examples(input_file):
        """Read a list of `InputExample`s from an input jsonl file,
        {"id": *str_id*, "text": *space separated sequence, tokenized, lowercased*}"""
        examples = []
        with open(input_file, "r", encoding='utf-8') as reader:
            while True:
                line = reader.readline()
                if not line:
                    break
                line = json.loads(line.strip())
                examples.append(
                    edict(unique_id=line["id"], text=line["text"], text_name=line["text_name"]))
        return examples

    def wordpiece_tokenize(self, tokens):
        """tokens (list of str): tokens from another tokenizer"""
        assert isinstance(tokens, list)
        wp_tokens = []
        wp_token_map = []
        for t in tokens:
            wp_token_map.append(len(wp_tokens))
            wp_tokens.extend(self.wordpiece_tokenizer.tokenize(t))
        return wp_tokens, wp_token_map

    def normalize_text(self, tokens, max_seq_length):
        """convert to a word piece tokenized list of tokens
        from pre-tokenized tokens"""
        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        wp_tokens, wp_token_map = self.wordpiece_tokenize(tokens)
        if len(wp_tokens) > max_seq_length - 2:
            wp_tokens = wp_tokens[0:(max_seq_length - 2)]
            max_idx_location = np.searchsorted(wp_token_map, max_seq_length - 2, side="left")  # right
            wp_token_map = wp_token_map[:max_idx_location]
        wp_tokens = ["[CLS]"] + wp_tokens + ["[SEP]"]
        wp_token_map = np.array(wp_token_map) + 1  # account for "[CLS]" token
        wp_token_map = wp_token_map.tolist()
        wp_token_map.append(-1)  # the last token is "[SEP]", which we will remove
        wp_token_ids = self.convert_tokens_to_ids(wp_tokens)
        return wp_tokens, wp_token_ids, wp_token_map

    def compare_tokenization(self, text):
        assert text.islower()
        original_tokens = text.split()

        wp_tokens, wp_token_ids, wp_map = self.normalize_text(original_tokens, self.max_seq_len)

        items = edict(
            tokens=wp_tokens,
            token_ids=wp_token_ids,
            tokens_internal=self.convert_ids_to_tokens(wp_token_ids),
            token_map=wp_map,
            original_tokens=original_tokens,
        )
        print("original tokens: {} \nwp tokens: {} \n internal tokens: {}".format(
            items.original_tokens, items.tokens, items.tokens_internal))
        return items


def load_qa_data():
    return load_json("./qa.json")

def load_sub_data():
    return load_json("./subtitle.json")

def mk_qa_input_lines(tvqa_data, unk):
    """tvqa_data list(dicts)"""
    lines = []
    keys = ["q", "a0", "a1", "a2", "a3", "a4"]  # , "sub_text"
    prefix = "s_tokenized_"  # stands for `stanford tokenized`
    for e in tvqa_data:
        qid = str(e["q_id"])
        for k in keys:
            text = e[prefix+k]
            for name in name_map.keys():
                if unk==True:
                    text = text.replace(name, '##name')
            lines.append(
                edict(
                    unique_id="{}_{}".format(qid, k),
                    text=text,
                    text_name=e[prefix+k])
            )
    return lines


def mk_sub_input_lines(tvqa_data, unk):
    # max len 500 100%, max len 400, 99.07%
    lines = {}  # with vid_names as keys
    for e in tvqa_data:
        vid_name = e["vid_name"]
        sub = e["s_tokenized_sub_text"].replace("<eos>", "\n").replace("UNKNAME", "##name")
        sub2 = e["s_tokenized_sub_text"].replace("<eos>", "\n").replace("UNKNAME", "##name")
        for name in name_map.keys():
            if unk==True:
                sub = sub.replace(name, '##name')
            
        sub_split = sub.split('\n')
        sub2_split = sub2.split('\n')
#        if vid_name not in lines:
        for i in range(len(sub_split)):
            lines[vid_name+str(i+10)] = (sub_split[i], sub2_split[i])

    lines = [edict(unique_id=k, text=v1, text_name=v2) for k, (v1,v2) in lines.items()]
    return lines


class CharacterNet(nn.Module):
    def __init__(self, bert_model_path):
        super(CharacterNet, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased').cuda()
        self.fc1 = nn.Linear(768,128)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(128,768)
        self.norm = Normalize()
    
    def get_original_token_embedding(self, padded_wp_token_embedding, wp_token_mask, token_map):
        """ subword embeddings from the same original token will be averaged to get
        the original token embedding.
        Args:
            padded_wp_token_embedding: (#wp_tokens, hsz)
            wp_token_mask: (#wp_tokens, )
            token_map (list of int): maps the word piece tokens to original tokens

        Returns:

        """
        unpadded_wp_embedding = padded_wp_token_embedding[:int(sum(wp_token_mask).item())]
        original_token_embedding = [unpadded_wp_embedding[token_map[i]:token_map[i+1]].mean(0)
                                    for i in range(len(token_map)-1)]
        return torch.stack(original_token_embedding)

    def forward(self, input_ids, input_mask, layer_index, token_map, unique_ids):
        all_encoder_layers, _ = self.bert(input_ids,
                                                token_type_ids=None,
                                                attention_mask=input_mask)
        all_encoder_layers = torch.stack(all_encoder_layers, dim = 0) # (12, batch_size, max_seq_length, 768)
        layer_output = all_encoder_layers[layer_index] # (batch_size, max_seq_length, 768)
        original_token_embeddings = torch.zeros((0,768), requires_grad=True).cuda()

        for batch_idx, unique_id in enumerate(unique_ids):
            original_token_embeddings = torch.cat((original_token_embeddings,self.get_original_token_embedding(layer_output[batch_idx], input_mask[batch_idx], token_map[batch_idx])), dim=0) # (sum_seq_length, 768)

        rep = self.fc1(original_token_embeddings) # (sum_seq_length, 1000)
        rep = self.relu(rep) # (sum_seq_length, 1000)
        rep = self.fc2(rep) # (sum_seq_length, 768)
        rep = self.norm(rep) # (sum_seq_length, 768)
        return rep, original_token_embeddings

class CharacterLoss(torch.nn.Module):

    def __init__(self, batch_size, use_youI):
        super(CharacterLoss, self).__init__()
        self.use_youI = use_youI
        self.batch_size = batch_size
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
        self.name_list = (load_json('./name.json')).keys()
        self.name_list_lower = [x.lower() for x in self.name_list]
    
    def find_pos_cur(self, token, i):
        while True:
            i = i - 1
            if i==-1:
                return -1
            if (token[i] in self.name_list_lower) and token[i+1]==':':
                return i

    def find_pos_prev(self, token, i):
        sw = 0
        while True:
            i = i - 1
            if i==-1:
                return -1
            if sw==1 and (token[i] in self.name_list_lower) and token[i+1]==':':
                return i
            if sw==0 and (token[i] in self.name_list_lower) and token[i+1]==':':
                sw = 1
            
    def forward(self, data, token):
        sim_matrix = self._cosine_similarity(data.unsqueeze(1), data.unsqueeze(0)) # (sum_seq_length, sum_seq_length)
        token_indicies = []
        if self.use_youI:
            for i in range(len(token)):
                if token[i]=='me' or token[i]=='i':
                    p = self.find_pos_cur(token, i)
                    if p>=0:
                        token[i] = token[p]
                if token[i]=='you':
                    p = self.find_pos_prev(token, i)
                    if p>=0:
                        token[i] = token[p]

        for i in range(len(data)):
            if token[i] in self.name_list_lower:
                token_indicies.append(i)


        if len(token_indicies)==0:
            return torch.tensor(0, dtype=torch.float32, requires_grad=True).cuda()
        target_matrix = torch.zeros((len(token_indicies), len(token_indicies)), requires_grad=True).cuda() # (num_character, num_character)
        sim_matrix_2 = torch.zeros((len(token_indicies), len(token_indicies)), requires_grad=True).cuda() # (num_character, num_character)
        for i in range(len(token_indicies)):
            for j in range(len(token_indicies)):
                sim_matrix_2[i][j] = sim_matrix[token_indicies[i]][token_indicies[j]]
                if token[token_indicies[i]] == token[token_indicies[j]]:
                    target_matrix[i][j] = 1.0
                else:
                    target_matrix[i][j] = 0.0

        #loss2 = torch.tensor(0, dtype=torch.float32, requires_grad=True).cuda()
        loss = self.criterion(sim_matrix_2, target_matrix)
        return loss
def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    # parser.add_argument("--mode", default="qa", type=str, help="encode qa or sub")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    # Other parameters
    # parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using uncased model.")
    # parser.add_argument("--layers", default="-2", type=str)
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences longer "
                             "than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--batch_size", type=int, help="Batch size for predictions.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--inference", action="store_true")
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--ratio", type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--use_youI", action="store_true")
    parser.add_argument("--save_path", type=str)

    args = parser.parse_args()
    model_path = str(args.save_path) + 'model14_'+str(args.ratio)+'_'+str(args.epoch)+'_'+str(args.batch_size)+'_'+str(args.use_youI)+'_base.pt'

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


    input_data_qa = mk_qa_input_lines(load_qa_data(), unk = False)
    input_data_sub = mk_sub_input_lines(load_sub_data(), unk = False)
    input_data_unk = mk_sub_input_lines(load_sub_data(), unk = False) #+ mk_qa_input_lines(load_qa_data(), unk = False)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
    logger.info("device: {} n_gpu: {}".format(device, n_gpu))

    # layer_indexes = [int(x) for x in args.layers.split(",")]
    layer_index = -2  # second-to-last, which showed reasonable performance in BERT paper

    dset = BertSingleSeqDataset(input_data_unk, args.bert_model, args.max_seq_length)

    dset_inference_qa = BertSingleSeqDataset(input_data_qa, args.bert_model, args.max_seq_length)
    dset_inference_sub = BertSingleSeqDataset(input_data_sub, args.bert_model, args.max_seq_length)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    train_sampler = SequentialSampler(dset)
    train_dataloader = DataLoader(dset, sampler=train_sampler, batch_size=args.batch_size,
                                 collate_fn=pad_collate, num_workers=8)
    qa_eval_sampler = SequentialSampler(dset_inference_qa)
    qa_eval_dataloader = DataLoader(dset_inference_qa, sampler=qa_eval_sampler, batch_size=args.batch_size,
                                 collate_fn=pad_collate, num_workers=8)
    sub_eval_sampler = SequentialSampler(dset_inference_sub)
    sub_eval_dataloader = DataLoader(dset_inference_sub, sampler=sub_eval_sampler, batch_size=args.batch_size,
                                 collate_fn=pad_collate, num_workers=8)
    cnt = 0
    sub_data = {}
    
    n_epoch = args.epoch
    
    net = CharacterNet(args.bert_model)
    net.cuda()
    loss_fn = CharacterLoss(batch_size = args.batch_size, use_youI = args.use_youI)
    optimizer = optim.SGD(net.parameters(), lr=0.0003, momentum=0.9)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=20, after_scheduler=optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=180))

    #Train & save model (sub+qa)
    
    if args.train:
        for epoch in range(1,n_epoch+1):
            epoch_start = time.time()
            train_loss = 0
            net.train()
            cc = 0
            for batch in tqdm(train_dataloader):
                cc = cc + 1
                optimizer.zero_grad()
                input_ids = batch.token_ids.cuda()
                input_mask = batch.token_ids_mask.cuda()
                unique_ids = batch.unique_id
                token_map = batch.token_map
                original_token = batch.original_tokens
                original_token_flatten = [item for sublist in original_token for item in sublist]
                representation , _ = net(input_ids, input_mask, layer_index, token_map, unique_ids)  # (#layers, bsz, #tokens, hsz)
                #print(representation)
                loss = loss_fn(representation, original_token_flatten)
                train_loss += loss.item()
                
                loss.backward()
                optimizer.step()
                if cc >= len(train_dataloader)*args.ratio/100:
                    break

                #print(train_loss)
            epoch_time = time.time() - epoch_start
            print("Epoch\t", epoch, "\tLoss\t", train_loss, "\tTime\t", epoch_time)
            scheduler.step()
        torch.save(net.state_dict(), model_path)

    
    #Inference & save embedding (sub, qa respectively)
    if args.inference:
        net = CharacterNet(args.bert_model)
        net.cuda()

        name_list = (load_json('./name.json')).keys()
        name_list_lower = [x.lower() for x in name_list]
        param = net.state_dict()
        data_list = np.zeros((0,768))
        token_list = []
        #net.load_state_dict(torch.load(model_path))
        net.eval()
        cc = 0
        sum1 = 0
        sum2 = 0
        cnt1 = 0
        cnt2 = 0
        for batch in tqdm(sub_eval_dataloader):
            input_ids = batch.token_ids.to(device)
            input_mask = batch.token_ids_mask.to(device)
            unique_ids = batch.unique_id
            token_map = batch.token_map
            original_token = batch.original_tokens
            _, original_token_embeddings= net(input_ids, input_mask, layer_index, token_map, unique_ids)
            original_token_embeddings = original_token_embeddings.cpu().detach().numpy()
            cur_len = 0
            #print(original_token_embeddings)
            for batch_idx, unique_id in enumerate(unique_ids):
                id = str(unique_id)[:-2]
                if id in sub_data:
                    sub_data[id] = np.concatenate([sub_data[id], original_token_embeddings[cur_len:cur_len+len(token_map[batch_idx])-1]], axis = 0).tolist()
                else:
                    sub_data[id] = original_token_embeddings[cur_len:cur_len+len(token_map[batch_idx])-1]
                cur_len = cur_len + len(token_map[batch_idx])-1
                
        h5_f = h5py.File(str(args.save_path) + 'sub14_'+str(args.ratio)+'_'+str(args.epoch)+'_'+str(args.batch_size)+'_'+str(args.use_youI)+'.h5', "w")
        for (k, v) in sub_data.items():
            h5_f.create_dataset(k, data=v, dtype=np.float32)
            
        h5_f = h5py.File(str(args.save_path) + 'qa14_'+str(args.ratio)+'_'+str(args.epoch)+'_'+str(args.batch_size)+'_'+str(args.use_youI)+'.h5', "w")
        for batch in tqdm(qa_eval_dataloader):
            input_ids = batch.token_ids.to(device)
            input_mask = batch.token_ids_mask.to(device)
            unique_ids = batch.unique_id
            token_map = batch.token_map
            original_token = batch.original_tokens
            _ , original_token_embeddings = net(input_ids, input_mask, layer_index, token_map, unique_ids)
            original_token_embeddings = original_token_embeddings.cpu().detach().numpy()

            cur_len = 0
            for batch_idx, unique_id in enumerate(unique_ids):
                h5_f.create_dataset(str(unique_id), data=original_token_embeddings[cur_len:cur_len+len(token_map[batch_idx])-1], dtype=np.float32)
                cur_len = cur_len + len(token_map[batch_idx])-1
        
if __name__ == "__main__":
    main()
