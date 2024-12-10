import torch
import argparse

from load_model import load_model
from evals import eval_perplexity
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
import torch.nn.functional as F
import sampling



def main():
    parser = argparse.ArgumentParser(description="Generate some samples")
    parser.add_argument("--model_path", default="louaaron/sedd-medium", type=str)
    parser.add_argument("--dataset", default="wikitext103", type=str)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=64)
    # parser.add_argument("--decoder", default="delta_diff")
    args = parser.parse_args()
    
    device = torch.device('cuda')
    model_med, graph_med, noise_med = load_model("louaaron/sedd-medium", device)
    model_small, graph_small, noise_small = load_model("louaaron/sedd-small", device)
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    baseline_perplexities = 0
    num_batches = 1
    total_perplexity = 0
    perplexities = []
    for i in range(num_batches):
        
        sampling_fn = sampling.get_contrastive_sampler(
            graph_med, graph_small, noise_med, noise_small,
            (args.batch_size, 256), 'analytic', args.steps, device=device,
            alpha = 0.2, sampling_method = "diff_max"
        )
        """
        sampling_fn = sampling.get_pc_sampler(
            graph, noise, (args.batch_size, 256), 'analytic', args.steps, device=device
        )
        """
        

        samples = sampling_fn(model_med, model_small)

        text_samples = tokenizer.batch_decode(samples)
        """
        for i in text_samples:
            print(i)
            print("=================================================")
        """
        perplexity, perplexity_list = eval_perplexity(samples, device)
        total_perplexity += perplexity
        perplexities.extend(perplexity_list)
    print(total_perplexity / num_batches)
    print(perplexities)

    

if __name__=="__main__":
    main()