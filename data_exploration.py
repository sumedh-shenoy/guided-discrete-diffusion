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
    
    """
    alphas = [0.1, 0.2, 0.25, 0.3, 0.5, 0.75, 1.0]
    
    with open('alpha_perplexities.pkl', 'rb') as file:
        stats = pickle.load(file)

    print(stats)

    for key, value in stats.items():
        if not isinstance(key, tuple) or key[1] != 64 or key[0] > 10:
            continue
        
        print(f"alpha {key[0]} gives perplexity of {value / 8}")
    """
    with open('timestep_perplexities.pkl', 'rb') as file:
        timestep_stats = pickle.load(file)

    for key, value in timestep_stats.items():
        print(f"{key} gives perplexity of {value / 8}")
    
"""
('med', 32) gives perplexity of 111.22921752929688
('small', 32) gives perplexity of 145.29393005371094
('new', 32) gives perplexity of 102.66381072998047
('med', 64) gives perplexity of 81.11799621582031
('small', 64) gives perplexity of 103.96752166748047
('new', 64) gives perplexity of 73.37916564941406
('med', 128) gives perplexity of 61.490028381347656
('small', 128) gives perplexity of 85.52437591552734
('new', 128) gives perplexity of 56.372493743896484
('med', 256) gives perplexity of 52.753807067871094
('small', 256) gives perplexity of 64.33465576171875
('new', 256) gives perplexity of 48.58473205566406
('med', 512) gives perplexity of 42.301361083984375
('small', 512) gives perplexity of 57.33761215209961
('new', 512) gives perplexity of 39.029781341552734
('med', 1024) gives perplexity of 36.034629821777344
('small', 1024) gives perplexity of 46.388031005859375
('new', 1024) gives perplexity of 32.82666778564453
"""

if __name__=="__main__":
    main()