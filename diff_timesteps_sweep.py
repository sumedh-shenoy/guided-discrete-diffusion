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
    
    num_steps = [32, 64, 128, 256, 512, 1024]
    #alphas = [0.25, 0.75, 1.0, 2.5, 5, 10, 15, 25, 50, 100]

    indiv_stats = {}
    cum_stats = {}

    device = torch.device('cuda')
    model_med, graph_med, noise_med = load_model("louaaron/sedd-medium", device)
    model_small, graph_small, noise_small = load_model("louaaron/sedd-small", device)
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

    for step_count in num_steps:
        cum_stats[("med", step_count)] = 0
        indiv_stats[("med", step_count)] = []

        cum_stats[("small", step_count)] = 0
        indiv_stats[("small", step_count)] = []

        cum_stats[("new", step_count)] = 0
        indiv_stats[("new", step_count)] = []


    num_batches = 8
    for step_count in num_steps:
        contrastive_sampler = sampling.get_contrastive_sampler(
            graph_med, graph_small, noise_med, noise_small,
            (128, 256), 'analytic', step_count, device=device, alpha = 0.25
        )

        med_sampler = sampling.get_pc_sampler(
            graph_med, noise_med, (128, 256), 'analytic', step_count, device=device
        )

        small_sampler = sampling.get_pc_sampler(
            graph_small, noise_small, (128, 256), 'analytic', step_count, device=device
        )


        total_perplexity = 0
        perplexities = []
        for i in range(num_batches):
            samples = contrastive_sampler(model_med, model_small)
            text_samples = tokenizer.batch_decode(samples)
            perplexity, perplexity_list = eval_perplexity(samples, device)
            total_perplexity += perplexity
            perplexities.extend(perplexity_list)
        cum_stats[("new", step_count)] += total_perplexity
        indiv_stats[("new", step_count)].extend(perplexities)
        
        total_perplexity = 0
        perplexities = []
        for i in range(num_batches):
            samples = med_sampler(model_med)
            text_samples = tokenizer.batch_decode(samples)
            perplexity, perplexity_list = eval_perplexity(samples, device)
            total_perplexity += perplexity
            perplexities.extend(perplexity_list)
        cum_stats[("med", step_count)] += total_perplexity
        indiv_stats[("med", step_count)].extend(perplexities)

        total_perplexity = 0
        perplexities = []
        for i in range(num_batches):
            samples = small_sampler(model_small)
            text_samples = tokenizer.batch_decode(samples)
            perplexity, perplexity_list = eval_perplexity(samples, device)
            total_perplexity += perplexity
            perplexities.extend(perplexity_list)
        cum_stats[("small", step_count)] += total_perplexity
        indiv_stats[("small", step_count)].extend(perplexities)

        with open('timestep_perplexities.pkl', 'wb') as file:
            pickle.dump(cum_stats, file)
            
        with open('timestep_perplexities_list.pkl', 'wb') as file3:
            pickle.dump(indiv_stats, file3)

    

if __name__=="__main__":
    main()