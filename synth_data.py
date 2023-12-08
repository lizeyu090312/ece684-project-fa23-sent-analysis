# Load model directly
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import set_seed

from tqdm import tqdm
import argparse
import os

import pandas as pd
import numpy as np
import torch

from nltk.tokenize import RegexpTokenizer


# total_samples = 50000
# num_beams = num_seqs = 100
# top_k = 500
# top_p = 0.7
# div_pen = 4.0
# repetition_penalty = 2.5
# temperature = 1.0


def write_data(positive):
    set_seed(np.random.randint(1, 1<<31))
    which_sent = rng.integers(0, high=int(len(data_dict[positive])))
    prompt_full = data_dict[positive][which_sent]
    which_interval = rng.integers(7, high=15, size=1)[0]
    if len(prompt_full) - which_interval > 0:
        which_start = rng.integers(0, high=len(prompt_full)-which_interval, size=1)[0]
    else:
        which_start = 0
        which_interval = len(prompt_full)
    
    print(which_start, which_interval)
    prompt = prompt_full[which_start:which_start+which_interval]
    prompt_len_char = len(' '.join(prompt)) + 1
    tokenizer = AutoTokenizer.from_pretrained(path_dict[positive])
    model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(path_dict[positive]).to(device)
    input_ids = tokenizer.encode(' '.join(prompt), return_tensors='pt').to(device)
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)

    output = model.generate(input_ids, max_length=256, num_return_sequences=args.num_beams, num_beams=args.num_beams, 
                            # top_k=args.top_k, top_p=args.top_p, 
                            num_beam_groups=args.num_beams, diversity_penalty=args.div_pen, temperature=args.temperature, 
                            early_stopping=False, 
                            repetition_penalty=args.repetition_penalty, attention_mask=attention_mask, no_repeat_ngram_size=2)
    for i, review in enumerate(output):
        generated_review = tokenizer.decode(review, skip_special_tokens=True)
        generated_review = token.tokenize(generated_review)
        # print(f"Generated review {i+1}: {generated_review}")
        f_ptr.write("\"%s\",%s\n" % (' '.join(generated_review[which_interval:]), pos_neg_dict[positive]))
        # f_ptr.write("\" %s \",%s\n" % (generated_review[prompt_len_whicchar:], pos_neg_dict[positive]))
    return


if __name__ == '__main__':

    f_name = 'IMDB_synthetic.csv'

    if f_name not in os.listdir('./'):
        f_ptr = open(f_name, 'w')
    else:
        f_ptr = open(f_name, 'a')
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--total_samples', type=int, required=True)
    parser.add_argument('-n', '--num_beams', type=int, required=True)
    # parser.add_argument('-k', '--top_k', type=int, required=True)
    # parser.add_argument('-p', '--top_p', type=float, required=True)
    parser.add_argument('-d', '--div_pen', type=float, required=True)
    parser.add_argument('-r', '--repetition_penalty', type=float, required=True)
    parser.add_argument('-f', '--temperature', type=float, required=True)
    parser.add_argument('-v', '--device', type=int, required=True)
    args = parser.parse_args()

    dataset = pd.read_csv('./IMDBDataset.csv')
    print(dataset)
    token = RegexpTokenizer(r'[a-zA-Z0-9]+')
    data_dict = {False: token.tokenize_sents(dataset.query('sentiment=="negative"')['review']), 
                 True: token.tokenize_sents(dataset.query('sentiment=="positive"')['review'])}

    path_dict = {True: "lvwerra/gpt2-imdb-pos", False: "mrm8488/gpt2-imdb-neg"}
    pos_neg_dict = {True: 'positive', False: 'negative'}
    # default usage: python synth_data.py -t 50000 -n 10 -d 20.0 -r 2.5 -f 1.0 -v 0 
    device = torch.device("cuda:%d" % args.device if torch.cuda.is_available() and args.device in {0, 1} else "cpu")

    rng = np.random.default_rng()
    for iter in tqdm(range(int(args.total_samples / args.num_beams / 2)), desc='Gen_Synthetic_Data'):
        write_data(True)
        write_data(False)