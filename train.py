import os
import pickle
import argparse
import torch
import random
import numpy as np
from tqdm import tqdm
from dataloader import TrajectoryDataset
from model import graph_tern
from metrics import mse_loss, gaussian_mixture_loss
from utils import data_sampler
from torch.utils.data import DataLoader


# Reproducibility
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# Argument parsing
parser = argparse.ArgumentParser()

# Model specific parameters
parser.add_argument('--input_size', type=int, default=2)
parser.add_argument('--output_size', type=int, default=5)
parser.add_argument('--n_epgcn', type=int, default=1, help='Number of EPGCN layers for endpoint prediction')
parser.add_argument('--n_epcnn', type=int, default=6, help='Number of EPCNN layers for endpoint prediction')
parser.add_argument('--n_trgcn', type=int, default=1, help='Number of TRGCN layers for trajectory refinement')
parser.add_argument('--n_trcnn', type=int, default=3, help='Number of TRCNN layers for trajectory refinement')
parser.add_argument('--n_ways', type=int, default=3, help='Number of control points for endpoint prediction')
parser.add_argument('--n_smpl', type=int, default=20, help='Number of samples for refine')
parser.add_argument('--kernel_size', type=int, default=3)

# Data specifc paremeters
parser.add_argument('--obs_seq_len', type=int, default=8)
parser.add_argument('--pred_seq_len', type=int, default=12)
parser.add_argument('--dataset', default='zara1', help='Dataset name(eth,hotel,univ,zara1,zara2)')

# Training specifc parameters
parser.add_argument('--batch_size', type=int, default=128, help='Mini batch size')
parser.add_argument('--num_epochs', type=int, default=512, help='Number of epochs')
parser.add_argument('--clip_grad', type=float, default=None, help='Gradient clipping')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--lr_sh_rate', type=int, default=128, help='Number of steps to drop the lr')
parser.add_argument('--use_lrschd', action="store_true", default=False, help='Use lr rate scheduler')
parser.add_argument('--tag', default='tag', help='Personal tag for the model')

args = parser.parse_args()

# Data preparation
# Batch size set to 1 because vertices vary by humans in each scene sequence.
# Use mini batch working like batch.
dataset_path = './datasets/' + args.dataset + '/'
checkpoint_dir = './checkpoint/' + args.tag + '/'

train_dataset = TrajectoryDataset(dataset_path + 'train/', obs_len=args.obs_seq_len, pred_len=args.pred_seq_len, skip=1)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)

val_dataset = TrajectoryDataset(dataset_path + 'val/', obs_len=args.obs_seq_len, pred_len=args.pred_seq_len, skip=1)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

# Model preparation
model = graph_tern(n_epgcn=args.n_epgcn, n_epcnn=args.n_epcnn, n_trgcn=args.n_trgcn, n_trcnn=args.n_trcnn,
                   seq_len=args.obs_seq_len, pred_seq_len=args.pred_seq_len, n_ways=args.n_ways, n_smpl=args.n_smpl)
model = model.cuda()

optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
if args.use_lrschd:
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_sh_rate, gamma=0.8)

# Train logging
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
with open(checkpoint_dir + 'args.pkl', 'wb') as f:
    pickle.dump(args, f)

metrics = {'train_loss': [], 'val_loss': []}
constant_metrics = {'min_val_epoch': -1, 'min_val_loss': 1e10}


def train(epoch):
    global metrics, model
    model.train()
    loss_batch = 0.
    r_loss_batch, m_loss_batch = 0., 0.
    loader_len = len(train_loader)

    progressbar = tqdm(range(loader_len))
    progressbar.set_description('Train Epoch: {0} Loss: {1:.8f}'.format(epoch, 0))
    optimizer.zero_grad()
    for batch_idx, batch in enumerate(train_loader):
        # sum gradients till idx reach to batch_size
        if batch_idx % args.batch_size == 0:
            optimizer.zero_grad()

        S_obs, S_trgt = [tensor.cuda() for tensor in batch[-2:]]

        # Data augmentation
        aug = True
        if aug:
            S_obs, S_trgt = data_sampler(S_obs, S_trgt, batch=1)

        # Run Graph-TERN model
        V_init, V_pred, V_refi, valid_mask = model(S_obs, S_trgt)

        # Loss calculation
        r_loss = gaussian_mixture_loss(V_init, S_trgt[:, 1], args.n_ways)
        m_loss = mse_loss(V_refi, S_trgt[:, 0], valid_mask)
        loss = r_loss + m_loss

        if torch.isnan(loss):
            pass
        else:
            loss.backward()
            loss_batch += loss.item()

        r_loss_batch += r_loss.item()
        m_loss_batch += m_loss.item()

        if batch_idx % args.batch_size + 1 == args.batch_size or batch_idx + 1 == loader_len:
            if args.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()

            r_loss_batch = 0.
            m_loss_batch = 0.

        progressbar.set_description('Train Epoch: {0} Loss: {1:.8f}'.format(epoch, loss.item() / args.batch_size))
        progressbar.update(1)

    progressbar.close()
    metrics['train_loss'].append(loss_batch / loader_len)


def valid(epoch):
    global metrics, constant_metrics, model
    model.eval()
    loss_batch = 0.
    r_loss_batch, m_loss_batch = 0., 0.
    loader_len = len(val_loader)

    progressbar = tqdm(range(loader_len))
    progressbar.set_description('Valid Epoch: {0} Loss: {1:.8f}'.format(epoch, 0))

    for batch_idx, batch in enumerate(val_loader):
        S_obs, S_trgt = [tensor.cuda() for tensor in batch[-2:]]

        # Run Graph-TERN model
        V_init, V_pred, V_refi, valid_mask = model(S_obs)

        # Loss calculation
        r_loss = gaussian_mixture_loss(V_init, S_trgt[:, 1], args.n_ways)
        m_loss = mse_loss(V_refi, S_trgt[:, 0], valid_mask, training=False)
        loss = r_loss + m_loss

        loss_batch += loss.item()
        r_loss_batch += r_loss.item()
        m_loss_batch += m_loss.item()

        if batch_idx % args.batch_size + 1 == args.batch_size or batch_idx + 1 == loader_len:
            r_loss_batch = 0.
            m_loss_batch = 0.

        progressbar.set_description('Valid Epoch: {0} Loss: {1:.8f}'.format(epoch, loss.item() / args.batch_size))
        progressbar.update(1)

    progressbar.close()
    metrics['val_loss'].append(loss_batch / loader_len)

    # Save model
    if metrics['val_loss'][-1] < constant_metrics['min_val_loss']:
        constant_metrics['min_val_loss'] = metrics['val_loss'][-1]
        constant_metrics['min_val_epoch'] = epoch
        torch.save(model.state_dict(), checkpoint_dir + args.dataset + '_best.pth')


def main():
    for epoch in range(args.num_epochs):
        train(epoch)
        valid(epoch)

        if args.use_lrschd:
            scheduler.step()

        print(" ")
        print("Dataset: {0}, Epoch: {1}".format(args.tag, epoch))
        print("Train_loss: {0}, Val_los: {1}".format(metrics['train_loss'][-1], metrics['val_loss'][-1]))
        print("Min_val_epoch: {0}, Min_val_loss: {1}".format(constant_metrics['min_val_epoch'], constant_metrics['min_val_loss']))
        print(" ")

        with open(checkpoint_dir + 'metrics.pkl', 'wb') as f:
            pickle.dump(metrics, f)

        with open(checkpoint_dir + 'constant_metrics.pkl', 'wb') as f:
            pickle.dump(constant_metrics, f)


if __name__ == "__main__":
    main()
