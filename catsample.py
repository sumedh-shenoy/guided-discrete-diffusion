import torch
import torch.nn.functional as F


def gumbel_softmax(categorical_probs, hard=False, eps=1e-9):
    logits = categorical_probs.clamp(min=1e-9).log()
    return F.gumbel_softmax(logits, hard=hard)


def sample_categorical(categorical_probs, method="hard"):
    if method == "hard":
        #print("triggered here")
        gumbel_norm = 1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log()
        return (categorical_probs / gumbel_norm).argmax(dim=-1)
    else:
        raise ValueError(f"Method {method} for sampling categorical variables is not valid.")

def sample_categorical_diff_delta(probs_med, probs_small, alpha):
    delta_prob = probs_med - probs_small
    #print(delta_prob.sum())
    probs_new = probs_med + alpha * delta_prob
    gumbel_norm = 1e-10 - (torch.rand_like(probs_new) + 1e-10).log()
    return (probs_new / gumbel_norm).argmax(dim=-1)

def sample_categorical_diff(probs_med, probs_small, alpha):
    probs_new = probs_med - alpha * probs_small
    gumbel_norm = 1e-10 - (torch.rand_like(probs_med) + 1e-10).log()
    return (probs_new / gumbel_norm).argmax(dim=-1)

def sample_categorical_logit_diff(probs_med, probs_small, alpha):
    med_logits = probs_med.log()
    print(probs_med.sum())
    small_logits = probs_small.log()
    new_logits = med_logits - alpha * small_logits
    new_probs = F.softmax(med_logits)

    return sample_categorical(new_probs)

def sample_categorical_logit_diff_delta(probs_med, probs_small, alpha):
    med_logits = probs_med.log()
    small_logits = probs_small.log()
    new_logits = med_logits + (med_logits - small_logits) * alpha
    new_probs = F.softmax(new_logits)

    return sample_categorical(new_probs)

def sample_categorical_logit_diff_max(probs_med, probs_small, alpha):
    threshold = alpha * probs_med.max()
    mask = (probs_med > threshold)
    med_logits = probs_med.log()
    small_logits = probs_small.log()
    new_logits = (med_logits - small_logits) * mask
    new_probs = F.softmax(new_logits)

    return sample_categorical(new_probs)

    