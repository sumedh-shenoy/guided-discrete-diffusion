import torch
import argparse

from load_model import load_model
from evals import eval_perplexity
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
import torch.nn.functional as F
import sampling



def main():
    parser = argparse.ArgumentParser(description="Generate some samples")
    parser.add_argument("--model_path", default="louaaron/sedd-small", type=str)
    parser.add_argument("--dataset", default="wikitext103", type=str)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=64)
    # parser.add_argument("--decoder", default="delta_diff")
    args = parser.parse_args()
    
    device = torch.device('cuda')
    model, graph, noise = load_model(args.model_path, device)
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    baseline_perplexity = 0
    num_batches = 8
    for i in range(num_batches):

        sampling_fn = sampling.get_pc_sampler(
            graph, noise, (args.batch_size, 256), 'analytic', args.steps, device=device
        )
        

        samples = sampling_fn(model)

        text_samples = tokenizer.batch_decode(samples)
        """
        for i in text_samples:
            print(i)
            print("=================================================")
        """
        perplexity = eval_perplexity(samples, device)
        baseline_perplexity += perplexity
    print(baseline_perplexity / num_batches)

    

if __name__=="__main__":
    main()