import torch
from torch.distributions import Categorical, Normal, Independent, MixtureSameFamily


def gaussian_mixture_loss(W_pred, S_trgt, n_stop):
    r"""Batch gaussian mixture loss"""
    # NMV(C*K) -> NVM(C*K)
    W_pred = W_pred.transpose(1, 2).contiguous()

    temp = S_trgt.chunk(chunks=n_stop, dim=1)
    W_trgt_list = [i.mean(dim=1) for i in temp]
    W_pred_list = W_pred.chunk(chunks=n_stop, dim=-1)

    loss_list = []
    for i in range(n_stop):
        # NVMC
        W_pred_one = W_pred_list[i]
        W_trgt_one = W_trgt_list[i]
        mix = Categorical(torch.nn.functional.softmax(W_pred_one[:, :, :, 4], dim=-1))
        comp = Independent(Normal(W_pred_one[:, :, :, 0:2], W_pred_one[:, :, :, 2:4].exp()), 1)
        gmm = MixtureSameFamily(mix, comp)
        loss_list.append(-gmm.log_prob(W_trgt_one))

    loss = torch.cat(loss_list, dim=0)
    return loss.mean()


def mse_loss(S_pred, S_trgt, loss_mask, training=True):
    r"""Batch mean square error loss"""
    # NTVC
    loss = (S_pred - S_trgt).norm(p=2, dim=3) ** 2
    loss = loss.mean(dim=1) * loss_mask
    if training:
        loss[loss > 1] = 0
    return loss.mean()
