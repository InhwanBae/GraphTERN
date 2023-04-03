import math
import random
import torch


def data_sampler(S_obs, S_trgt, batch=4, scale=True, stretch=False, flip=True, rotation=True, noise=False):
    r"""Returns the Trajectories with batch size."""

    aug_So, aug_Sg = [], []

    for i in range(batch):
        S_obs_t, S_tr_t = S_obs.clone(), S_trgt.clone()
        if scale:
            S_obs_t, S_tr_t = random_scale(S_obs_t, S_tr_t, min=0.8, max=1.2)
        if stretch:
            S_obs_t, S_tr_t = random_stretch(S_obs_t, S_tr_t, min=0.8, max=1.2)
        if flip:
            S_obs_t, S_tr_t = random_flip(S_obs_t, S_tr_t)
        if rotation:
            S_obs_t, S_tr_t = random_rotation(S_obs_t, S_tr_t)
        if noise:
            S_obs_t, S_tr_t = random_noise(S_obs_t, S_tr_t)

        aug_So.append(S_obs_t.squeeze(dim=0))
        aug_Sg.append(S_tr_t.squeeze(dim=0))

    S_obs = torch.stack(aug_So).detach()
    S_trgt = torch.stack(aug_Sg).detach()

    return S_obs, S_trgt


def random_scale(S_obs, S_trgt, min=0.8, max=1.2):
    r"""Returns the randomly scaled Trajectories."""

    scale = random.uniform(min, max)
    return S_obs * scale, S_trgt * scale


def random_stretch(S_obs, S_trgt, min=0.9, max=1.1):
    r"""Returns the randomly stretched Trajectories."""

    scale = [random.uniform(min, max), random.uniform(min, max)]
    scale = torch.tensor(scale).cuda()
    scale_a = torch.sqrt(scale[0] * scale[1])
    return S_obs * scale, S_trgt * scale


def random_flip(S_obs, S_trgt):
    r"""Returns the randomly flipped Trajectories."""

    flip = random.choice([[-1, -1], [-1, 1], [1, -1], [1, 1]])
    flip = torch.tensor(flip).cuda()
    return S_obs * flip, S_trgt * flip


def random_rotation(S_obs, S_trgt):
    r"""Returns the randomly rotated Trajectories."""

    theta = random.uniform(-math.pi, math.pi)
    # for 90 degree augmentation
    theta = (theta // (math.pi/2)) * (math.pi/2)

    r_mat = [[math.cos(theta), -math.sin(theta)],
             [math.sin(theta), math.cos(theta)]]
    r = torch.tensor(r_mat, dtype=torch.float, requires_grad=False).cuda()

    S_obs = torch.einsum('rc,natvc->natvr', r, S_obs)
    S_trgt = torch.einsum('rc,natvc->natvr', r, S_trgt)
    return S_obs, S_trgt


def random_noise(S_obs, S_trgt, std=0.01):
    r"""Returns the randomly noised Trajectories."""

    noise_obs = torch.randn_like(S_obs) * std
    noise_tr = torch.randn_like(S_trgt) * std
    return S_obs + noise_obs, S_trgt + noise_tr
