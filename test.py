import pickle
import argparse
import torch
import numpy as np
from tqdm import tqdm
from graphtern.model import graph_tern
from utils.dataloader import TrajectoryDataset
from torch.utils.data import DataLoader


# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--tag', default='tag', help='Personal tag for the model')
parser.add_argument('--n_samples', type=int, default=20, help='Number of samples')
test_args = parser.parse_args()

# Get arguments for training
checkpoint_dir = './checkpoint/' + test_args.tag + '/'

args_path = checkpoint_dir + '/args.pkl'
with open(args_path, 'rb') as f:
    args = pickle.load(f)

dataset_path = './datasets/' + args.dataset + '/'
model_path = checkpoint_dir + args.dataset + '_best.pth'

# Data preparation
test_dataset = TrajectoryDataset(dataset_path + 'test/', obs_len=args.obs_seq_len, pred_len=args.pred_seq_len, skip=1)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

# Model preparation
model = graph_tern(n_epgcn=args.n_epgcn, n_epcnn=args.n_epcnn, n_trgcn=args.n_trgcn, n_trcnn=args.n_trcnn,
                   seq_len=args.obs_seq_len, pred_seq_len=args.pred_seq_len, n_ways=args.n_ways, n_smpl=args.n_smpl)
model = model.cuda()
model.load_state_dict(torch.load(model_path), strict=False)


def test(KSTEPS=20):
    model.eval()
    model.n_smpl = KSTEPS
    ade_refi_all = []
    fde_refi_all = []

    progressbar = tqdm(range(len(test_loader)))
    progressbar.set_description('Testing {}'.format(test_args.tag))

    for batch_idx, batch in enumerate(test_loader):
        S_obs, S_trgt = [tensor.cuda() for tensor in batch[-2:]]

        # Run Graph-TERN model
        V_init, V_pred, V_refi, valid_mask = model(S_obs, pruning=4, clustering=True)

        # Calculate ADEs and FDEs for each refined trajectory
        V_trgt_abs = S_trgt[:, 0].squeeze(dim=0)
        temp = (V_refi - V_trgt_abs).norm(p=2, dim=-1)
        ADEs = temp.mean(dim=1).min(dim=0)[0]
        FDEs = temp[:, -1, :].min(dim=0)[0]
        ade_refi_all.extend(ADEs.tolist())
        fde_refi_all.extend(FDEs.tolist())

        progressbar.update(1)

    progressbar.close()

    ade_refi = sum(ade_refi_all) / len(ade_refi_all)
    fde_refi = sum(fde_refi_all) / len(fde_refi_all)
    return ade_refi, fde_refi


def main():
    ade_refi, fde_refi = [], []

    # Repeat the evaluation to reduce randomness
    repeat = 10
    for i in range(repeat):
        temp = test(KSTEPS=test_args.n_samples)
        ade_refi.append(temp[0])
        fde_refi.append(temp[1])

    ade_refi = np.mean(ade_refi)
    fde_refi = np.mean(fde_refi)

    result_lines = ["Evaluating model: {}".format(test_args.tag),
                    "Refined_ADE: {0:.8f}, Refined_FDE: {1:.8f}".format(ade_refi, fde_refi)]

    for line in result_lines:
        print(line)


if __name__ == "__main__":
    main()
