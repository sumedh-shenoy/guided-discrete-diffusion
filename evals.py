import torch
import argparse

from load_model import load_model
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
import torch.nn.functional as F
import sampling

def eval_perplexity(sample, device="cpu"):
    perplexity_batch_size = 1
    with torch.no_grad():
        perplexities = []
        eval_model = GPT2LMHeadModel.from_pretrained("gpt2-large").to(device).eval()
        batches = sample.shape[0] // perplexity_batch_size
        total_perplexity = 0
        for i in range(batches):
            s = sample[i * perplexity_batch_size:(i + 1) * perplexity_batch_size]
            loss, logits = eval_model(s, labels=s)[:2]
            logits = logits.transpose(-1, -2)
            perplexity = F.cross_entropy(logits[..., :-1], s[..., 1:], reduction="none").mean(dim=-1).exp().mean()
            total_perplexity += perplexity
            perplexities.append(perplexity.detach().cpu())
        total_perplexity /= batches
        print(f"Perplexity: {total_perplexity:.3f}.")

        del eval_model, logits, loss
        return total_perplexity, perplexities