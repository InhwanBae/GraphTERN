import math
import random
import torch
import numpy as np

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
np.seterr(divide='ignore', invalid='ignore')


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


def drop_edge(A, percent, training=True, inplace=False):
    r"""Returns the randomly dropped edge Adjacency matrix with preserve rate."""

    assert 0 <= percent <= 1.0
    if not training:
        return A
    A_prime = torch.rand_like(A)
    A_drop = A if inplace else A.clone()
    A_drop[A_prime > percent] = 0
    return A_drop


def normalized_adjacency_matrix(A):
    r"""Returns the normalized Adjacency matrix. D^-1/2 @ A @ D^-1/2"""

    node_degrees = A.sum(-1).unsqueeze(dim=-1)
    degs_inv_sqrt = torch.pow(node_degrees, -0.5)
    degs_inv_sqrt[torch.isinf(degs_inv_sqrt)] = 0
    norm_degs_matrix = torch.eye(A.size(-1)).cuda() * degs_inv_sqrt
    return norm_degs_matrix @ A @ norm_degs_matrix


def normalized_adjacency_tilde_matrix(A):
    r"""Returns the normalized Adjacency tilde (A~) matrix."""

    A_t = A + torch.eye(A.size(-1)).cuda()
    return normalized_adjacency_matrix(A_t)


def normalized_laplacian_matrix(A):
    r"""Returns the normalized Laplacian matrix."""

    return torch.eye(A.size(-1)).cuda() - normalized_adjacency_matrix(A)


def normalized_laplacian_tilde_matrix(A):
    r"""Returns the normalized Laplacian tilde (L~) matrix."""

    A_t = A + torch.eye(A.size(-1)).cuda()
    return torch.eye(A_t.size(-1)).cuda() - normalized_adjacency_matrix(A_t)


def trajectory_visualizer(V_pred, V_obs, V_trgt):
    r"""Visualize trajectories"""
    # trajectory_visualizer(V_refi.detach(), S_obs, S_trgt)
    import matplotlib.pyplot as plt

    # generate gt trajectory
    V_gt = torch.cat((V_obs[:, 0], V_trgt[:, 0]), dim=1).squeeze(dim=0)
    V_absl = V_pred

    # visualize trajectories
    V_absl_temp = V_absl.view(-1, V_absl.size(2), 2)[:, :, :].cpu().numpy()
    V_gt_temp = V_gt[:, :, :].cpu().numpy()

    V_pred_traj_gt = V_trgt[:, 0].squeeze(dim=0)
    temp = V_absl - V_pred_traj_gt
    temp = (temp ** 2).sum(dim=-1).sqrt()

    V_absl = V_absl.cpu().numpy()
    bestADETrajectory = temp.mean(dim=1).min(dim=0)[1].cpu().numpy()

    # Visualize trajectories
    linew = 3
    fig = plt.figure(figsize=(10, 8))

    for n in range(V_pred.size(2)):
        plt.plot(V_gt_temp[:8, n, 0], V_gt_temp[:8, n, 1], linestyle='-', color='darkorange', linewidth=linew)
        plt.plot(V_gt_temp[7:, n, 0], V_gt_temp[7:, n, 1], linestyle='-', color='lime', linewidth=linew)

        bestTrajectory = V_absl[bestADETrajectory[n], :, n]
        plt.plot([V_gt_temp[7, n, 0], bestTrajectory[0, 0]], [V_gt_temp[7, n, 1], bestTrajectory[0, 1]], linestyle='-',
                 color='yellow', linewidth=linew)
        plt.plot(bestTrajectory[:, 0], bestTrajectory[:, 1], linestyle='-', color='yellow', linewidth=linew)

    plt.tick_params(axis="y", direction="in", pad=-22)
    plt.tick_params(axis="x", direction="in", pad=-15)
    plt.xlim(-11, 18)
    plt.ylim(-12, 15)
    plt.tight_layout()

    plt.show()


def controlpoint_visualizer(V_pred, samples=1000, n_levels=10):
    r"""Visualize control points"""
    # controlpoint_visualizer(V_init.detach())
    import matplotlib.pyplot as plt
    import seaborn as sns
    from torch.distributions import Categorical, Independent, Normal, MixtureSameFamily

    # NMV(C*K) -> NVM(C*K)
    n_stop = 3
    V_pred = V_pred.transpose(1, 2).contiguous()
    V_pred_list = V_pred.chunk(chunks=n_stop, dim=-1)

    # Generate Gaussian Mixture Model
    V_smpl_list = []
    for i in range(n_stop):
        V_pred_one = V_pred_list[i]
        mix = Categorical(torch.ones_like(V_pred_one[:, :, :, 4]))
        comp = Independent(Normal(V_pred_one[:, :, :, 0:2], V_pred_one[:, :, :, 2:4].exp()), 1)
        gmm = MixtureSameFamily(mix, comp)
        V_smpl_list.append(gmm.sample((samples,)))
    V_smpl = torch.cat(V_smpl_list, dim=1) * (12 // n_stop)

    # Visualize control points
    fig = plt.figure(figsize=(10, 3))

    for i in range(n_stop):
        plt.subplot(1, 3, (i + 1))
        V_absl_temp = V_smpl[:, i].cpu().numpy()
        for n in range(V_smpl.size(2)):
            ax = sns.kdeplot(V_absl_temp[:, n, 0], V_absl_temp[:, n, 1], n_levels=n_levels, shade=True, thresh=0.5)
            ax.text(2.5, 2.5, r'$\hat{c}_{' + str(i + 1) + r'}$', fontsize=16)

        ax.tick_params(axis="y", direction="in", pad=-22)
        ax.tick_params(axis="x", direction="in", pad=-15)
        plt.xlim(-3.5, 3.5)
        plt.ylim(-3.5, 3.5)

    plt.tight_layout()
    plt.show()
