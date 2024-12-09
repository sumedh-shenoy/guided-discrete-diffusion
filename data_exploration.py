import torch
import argparse

from load_model import load_model
from evals import eval_perplexity
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
import torch.nn.functional as F
import sampling
import pickle



def main():
    parser = argparse.ArgumentParser(description="Generate some samples")
    parser.add_argument("--dataset", default="wikitext103", type=str)
    parser.add_argument("--steps", type=int, default=1024)
    # parser.add_argument("--decoder", default="delta_diff")
    args = parser.parse_args()
    
    alphas = [0.1, 0.2, 0.25, 0.3, 0.5, 0.75, 1.0]
    
    with open('alpha_perplexities.pkl', 'rb') as file:
        stats = pickle.load(file)

    print(stats)

    for key, value in stats.items():
        if not isinstance(key, tuple) or key[1] != 64 or key[0] > 10:
            continue
        
        print(f"alpha {key} gives perplexity of {value / 8}")

    

if __name__=="__main__":
    main()