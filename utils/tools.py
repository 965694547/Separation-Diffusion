import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def vpsde_beta_t(t, T, min_beta, max_beta):
    t_coef = (2 * t - 1) / (T ** 2)
    return 1. - np.exp(-min_beta / T - 0.5 * (max_beta - min_beta) * t_coef)


def get_noise_schedule_list(schedule_mode, timesteps, min_beta=0.0, max_beta=0.01, s=0.008):
    if schedule_mode == "linear":
        schedule_list = np.linspace(1e-4, max_beta, timesteps)
    elif schedule_mode == "cosine":
        steps = timesteps + 1
        x = np.linspace(0, steps, steps)
        alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        schedule_list = np.clip(betas, a_min=0, a_max=0.999)
    elif schedule_mode == "vpsde":
        schedule_list = np.array([
            vpsde_beta_t(t, timesteps, min_beta, max_beta) for t in range(1, timesteps + 1)])
    else:
        raise NotImplementedError
    return torch.Tensor(schedule_list)


def calc_diffusion_hyperparams(schedule_mode, T, beta_0, beta_T, s):
    Beta = get_noise_schedule_list(schedule_mode, T, beta_0, beta_T, s)
    Alpha = 1 - Beta
    Alpha_bar = Alpha + 0
    Beta_tilde = Beta + 0
    for t in range(1, T):
        Alpha_bar[t] *= Alpha_bar[t - 1]
        Beta_tilde[t] *= (1 - Alpha_bar[t - 1]) / (
                    1 - Alpha_bar[t])
    Sigma = torch.sqrt(Beta_tilde)

    _dh = {}
    _dh["T"], _dh["Beta"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Sigma"] = T, Beta, Alpha, Alpha_bar, Sigma
    _dh["Alpha_bar"] = torch.cat(
        (
            torch.ones(1),
            _dh["Alpha_bar"]
         )
    )
    diffusion_hyperparams = _dh
    if diffusion_hyperparams != None:
        for key in diffusion_hyperparams:
            if key != "T":
                diffusion_hyperparams[key] = diffusion_hyperparams[key].to(device)
    return diffusion_hyperparams


def std_normal(size, num_spks):
    return torch.normal(0, 1, size=size).repeat(1, 1, num_spks).to(device)


def transformer_source(diffusion_steps, diffusion_hyperparams, source, z=None):
    steps = (1 + diffusion_steps)
    per = diffusion_hyperparams["Alpha_bar"][steps].to(device)
    if z == None:
        z = std_normal(source.shape)
    transformed_X = torch.sqrt(per) * source + torch.sqrt(1 - per) * z
    return transformed_X


def calc_diffusion_step_embedding(diffusion_steps, diffusion_step_embed_dim_in):
    assert diffusion_step_embed_dim_in % 2 == 0

    half_dim = diffusion_step_embed_dim_in // 2
    _embed = np.log(10000) / (half_dim - 1)
    _embed = torch.exp(torch.arange(half_dim) * -_embed)
    _embed = diffusion_steps * _embed
    diffusion_step_embed = torch.cat((torch.sin(_embed),
                                      torch.cos(_embed)), 1)

    return diffusion_step_embed.to(device)


def process_f0(f0, hparams):
    f0_ = (f0 - hparams['f0_mean']) / hparams['f0_std']
    f0_[f0 == 0] = np.interp(np.where(f0 == 0)[0], np.where(f0 > 0)[0], f0_[f0 > 0])
    uv = (torch.FloatTensor(f0) == 0).float()
    f0 = f0_
    f0 = torch.FloatTensor(f0)
    return f0, uv