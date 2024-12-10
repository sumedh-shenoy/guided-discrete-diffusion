import abc
import torch
import torch.nn.functional as F
import catsample

from model import utils as mutils

_PREDICTORS = {}


def register_predictor(cls=None, *, name=None):
    """A decorator for registering predictor classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _PREDICTORS:
            raise ValueError(
                f'Already registered model with name: {local_name}')
        _PREDICTORS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)

    
def get_predictor(name):
    return _PREDICTORS[name]



class Predictor(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(self, graph, noise):
        super().__init__()
        self.graph = graph
        self.noise = noise

    @abc.abstractmethod
    def update_fn(self, score_fn, x, t, step_size):
        """One update of the predictor.

        Args:
            score_fn: score function
            x: A PyTorch tensor representing the current state
            t: A Pytorch tensor representing the current time step.

        Returns:
            x: A PyTorch tensor of the next state.
        """
        pass

class PredictorContrastive(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(self, graph_med, noise_med, graph_small, noise_small):
        super().__init__()
        self.graph_med = graph_med
        self.noise_med = noise_med
        self.graph_small = graph_small
        self.noise_small = noise_small

    @abc.abstractmethod
    def update_fn(self, score_fn, x, t, step_size):
        """One update of the predictor.

        Args:
            score_fn: score function
            x: A PyTorch tensor representing the current state
            t: A Pytorch tensor representing the current time step.

        Returns:
            x: A PyTorch tensor of the next state.
        """
        pass


@register_predictor(name="euler")
class EulerPredictor(Predictor):
    def update_fn(self, score_fn, x, t, step_size):
        sigma, dsigma = self.noise(t)
        score = score_fn(x, sigma)

        rev_rate = step_size * dsigma[..., None] * self.graph.reverse_rate(x, score)
        x = self.graph.sample_rate(x, rev_rate)
        return x

@register_predictor(name="none")
class NonePredictor(Predictor):
    def update_fn(self, score_fn, x, t, step_size):
        return x


@register_predictor(name="analytic")
class AnalyticPredictor(Predictor):
    def update_fn(self, score_fn, x, t, step_size):
        curr_sigma = self.noise(t)[0]
        next_sigma = self.noise(t - step_size)[0]
        dsigma = curr_sigma - next_sigma

        score = score_fn(x, curr_sigma)

        stag_score = self.graph.staggered_score(score, dsigma)
        probs = stag_score * self.graph.transp_transition(x, dsigma)
        return catsample.sample_categorical(probs)

    
class Denoiser:
    def __init__(self, graph, noise):
        self.graph = graph
        self.noise = noise

    def update_fn(self, score_fn, x, t):
        sigma = self.noise(t)[0]

        score = score_fn(x, sigma)
        stag_score = self.graph.staggered_score(score, sigma)
        probs = stag_score * self.graph.transp_transition(x, sigma)
        # truncate probabilities
        if self.graph.absorb:
            probs = probs[..., :-1]
        
        #return probs.argmax(dim=-1)
        return catsample.sample_categorical(probs)

class ContrastivePredictor:
    def __init__(self, graph_med, noise_med, graph_small, noise_small, alpha = 0.25, sampling_method = "diff_delta"):
        self.graph_med = graph_med
        self.noise_med = noise_med
        self.graph_small = graph_small
        self.noise_small = noise_small
        self.alpha = alpha
        self.sampling_method = sampling_method

    def update_fn(self, score_fn_med, score_fn_small, x, t, step_size):
        curr_sigma_med = self.noise_med(t)[0]
        next_sigma_med = self.noise_med(t - step_size)[0]
        dsigma_med = curr_sigma_med - next_sigma_med

        curr_sigma_small = self.noise_small(t)[0]
        next_sigma_small = self.noise_med(t - step_size)[0]
        dsigma_small = curr_sigma_small - next_sigma_small

        score_med = score_fn_med(x, curr_sigma_med)
        stag_score_med = self.graph_med.staggered_score(score_med, dsigma_med)

        score_small = score_fn_small(x, curr_sigma_small)
        stag_score_small = self.graph_small.staggered_score(score_small, dsigma_small)

        probs_med = stag_score_med * self.graph_med.transp_transition(x, dsigma_med)
        probs_small = stag_score_small * self.graph_small.transp_transition(x, dsigma_small)

        if self.sampling_method == "diff_max":
            return catsample.sample_categorical_diff_max(probs_med, probs_small, self.alpha)
        return catsample.sample_categorical_diff_delta(probs_med, probs_small, self.alpha)



class ContrastiveDenoiser:
    def __init__(self, graph_med, noise_med, graph_small, noise_small, alpha = 0.25, sampling_method = "diff_delta"):
        self.graph_med = graph_med
        self.noise_med = noise_med
        self.graph_small = graph_small
        self.noise_small = noise_small
        self.alpha = alpha
        self.sampling_method = sampling_method

    def update_fn(self, score_fn_med, score_fn_small, x, t):
        sigma_med = self.noise_med(t)[0]
        sigma_small = self.noise_small(t)[0]

        score_med = score_fn_med(x, sigma_med)
        stag_score_med = self.graph_med.staggered_score(score_med, sigma_med)
        probs_med = stag_score_med * self.graph_med.transp_transition(x, sigma_med)

        score_small = score_fn_small(x, sigma_small)
        stag_score_small = self.graph_small.staggered_score(score_small, sigma_small)
        probs_small = stag_score_small * self.graph_small.transp_transition(x, sigma_small)

        if self.graph_med.absorb:
            probs_med = probs_med[..., :-1]
        if self.graph_small.absorb:
            probs_small = probs_small[..., :-1]

        if self.sampling_method == "diff_max":
            return catsample.sample_categorical_diff_max(probs_med, probs_small, self.alpha)

        return catsample.sample_categorical_diff_delta(probs_med, probs_small, self.alpha)
                       

def get_sampling_fn(config, graph, noise, batch_dims, eps, device):
    
    sampling_fn = get_pc_sampler(graph=graph,
                                 noise=noise,
                                 batch_dims=batch_dims,
                                 predictor=config.sampling.predictor,
                                 steps=config.sampling.steps,
                                 denoise=config.sampling.noise_removal,
                                 eps=eps,
                                 device=device)
    
    return sampling_fn
    

def get_contrastive_sampler(graph_med, graph_small, noise_med, noise_small, 
                            batch_dims, predictor, steps, denoise=True, eps=1e-5,
                            device=torch.device('cpu'), proj_fun=lambda x: x, alpha = 0.25,
                            sampling_method = "diff_delta"):
    med_predictor = get_predictor('analytic')(graph_med, noise_med)
    predictor_combined = ContrastivePredictor(graph_med, noise_med, graph_small, noise_small, alpha = alpha, sampling_method = sampling_method)
    projector = proj_fun
    denoiser_combined = ContrastiveDenoiser(graph_med, noise_med, graph_small, noise_small, alpha = alpha, sampling_method = sampling_method)

    @torch.no_grad()
    def pc_sampler(model_med, model_small):
        sampling_score_fn_med = mutils.get_score_fn(model_med, train=False, sampling=True)
        sampling_score_fn_small = mutils.get_score_fn(model_small, train=False, sampling=True)

        x = graph_med.sample_limit(*batch_dims).to(device)
        timesteps = torch.linspace(1, eps, steps + 1, device=device)
        dt = (1 - eps) / steps

        for i in range(steps):
            t = timesteps[i] * torch.ones(x.shape[0], 1, device=device)
            x = projector(x)
            x = predictor_combined.update_fn(sampling_score_fn_med, sampling_score_fn_small, x, t, dt)

        if denoise:
            x = projector(x)
            t = timesteps[-1] * torch.ones(x.shape[0], 1, device=device)
            x = denoiser_combined.update_fn(sampling_score_fn_med, sampling_score_fn_small, x, t)

        return x

    return pc_sampler

def get_contrastive_sampler_noised_normally(graph_med, graph_small, noise_med, noise_small, 
                            batch_dims, predictor, steps, denoise=True, eps=1e-5,
                            device=torch.device('cpu'), proj_fun=lambda x: x, alpha = 0.25):
    predictor_med = get_predictor(predictor)(graph_med, noise_med)
    predictor_combined = ContrastivePredictor(graph_med, noise_med, graph_small, noise_small, alpha = alpha)
    projector = proj_fun
    denoiser_combined = ContrastiveDenoiser(graph_med, noise_med, graph_small, noise_small, alpha = alpha)

    @torch.no_grad()
    def pc_sampler(model_med, model_small):
        sampling_score_fn_med = mutils.get_score_fn(model_med, train=False, sampling=True)
        sampling_score_fn_small = mutils.get_score_fn(model_small, train=False, sampling=True)

        x = graph_med.sample_limit(*batch_dims).to(device)
        timesteps = torch.linspace(1, eps, steps + 1, device=device)
        dt = (1 - eps) / steps

        for i in range(steps):
            t = timesteps[i] * torch.ones(x.shape[0], 1, device=device)
            x = projector(x)
            #x = predictor_combined.update_fn(sampling_score_fn_med, sampling_score_fn_small, x, t, dt)
            x = predictor_med.update_fn(sampling_score_fn_med, x, t, dt)

        if denoise:
            x = projector(x)
            t = timesteps[-1] * torch.ones(x.shape[0], 1, device=device)
            x = denoiser_combined.update_fn(sampling_score_fn_med, sampling_score_fn_small, x, t)

        return x

    return pc_sampler

def get_pc_sampler(graph, noise, batch_dims, predictor, steps, denoise=True, eps=1e-5, device=torch.device('cpu'), proj_fun=lambda x: x):
    predictor = get_predictor(predictor)(graph, noise)
    projector = proj_fun
    denoiser = Denoiser(graph, noise)

    @torch.no_grad()
    def pc_sampler(model):
        sampling_score_fn = mutils.get_score_fn(model, train=False, sampling=True)
        x = graph.sample_limit(*batch_dims).to(device)
        timesteps = torch.linspace(1, eps, steps + 1, device=device)
        dt = (1 - eps) / steps

        for i in range(steps):
            t = timesteps[i] * torch.ones(x.shape[0], 1, device=device)
            x = projector(x)
            x = predictor.update_fn(sampling_score_fn, x, t, dt)
            

        if denoise:
            # denoising step
            x = projector(x)
            t = timesteps[-1] * torch.ones(x.shape[0], 1, device=device)
            x = denoiser.update_fn(sampling_score_fn, x, t)
            
        return x
    
    return pc_sampler

