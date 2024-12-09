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
    
    alphas = [0.1, 0.2, 0.25, 0.3, 0.5, 0.75, 1.0, 2.0, 2.5, 5.0, 10.0]
    #alphas = [0.25, 0.75, 1.0, 2.5, 5, 10, 15, 25, 50, 100]

    with open('alpha_perplexities.pkl', 'rb') as file:
        cum_stats = pickle.load(file)

    indiv_stats = {}

    device = torch.device('cuda')
    model_med, graph_med, noise_med = load_model("louaaron/sedd-medium", device)
    model_small, graph_small, noise_small = load_model("louaaron/sedd-small", device)
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')


    steps = [64]

    for step_count in steps:
        for alpha_val in alphas:
            cum_stats[(alpha_val, step_count)] = 0
            indiv_stats[(alpha_val, step_count)] = []

    

    for step_count in steps:
        for alpha_val in alphas:
            for runs in range(8):
                sampling_fn = sampling.get_contrastive_sampler(
                    graph_med, graph_small, noise_med, noise_small,
                    (128, 256), 'analytic', step_count, device=device, alpha = alpha_val
                )

                samples = sampling_fn(model_med, model_small)

                text_samples = tokenizer.batch_decode(samples)
                # sample_dict[(alpha_val, step_count)] = sample_dict[(alpha_val, step_count)].extend(text_samples)
                """
                for i in text_samples:
                    print(i)
                    print("=================================================")
                """
                print(f"alpha of {alpha_val} and {step_count} steps")
                generative_perplexity, perplexity_list = eval_perplexity(samples, device)
                cum_stats[(alpha_val, step_count)] += generative_perplexity 
                indiv_stats[(alpha_val, step_count)].extend(perplexity_list)
            
            with open('alpha_perplexities.pkl', 'wb') as file:
                pickle.dump(cum_stats, file)
                
            with open('alpha_perplexities_list.pkl', 'wb') as file3:
                pickle.dump(indiv_stats, file3)

    

if __name__=="__main__":
    main()