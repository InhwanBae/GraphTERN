import torch
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
np.seterr(divide='ignore', invalid='ignore')


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
