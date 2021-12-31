import os
import torch
import time
import argparse
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from Discriminator import NPatchDiscriminators
from Generator import ResUNet
from GANLoss import GANLoss
from Hparams import HyperParams
from Data_utils import MelLoader, MelCollate
from Utils import build_D_tgt_grids, weights_init

def init_distributed(hparams, rank):
    torch.cuda.set_device(rank)
    dist.init_process_group(backend='nccl', init_method=hparams.dist_filepath, world_size=hparams.n_gpus, rank=rank)


def prepare_dataloaders(hparams):
    trainset = MelLoader(hparams.train_inputs_path, hparams.train_targets_path, hparams.train_identity)
    valset = MelLoader(hparams.test_inputs_path, hparams.test_targets_path, hparams.train_identity)
    collate_fn = MelCollate()

    train_loader = DataLoader(trainset, 
                              shuffle=True,
                              batch_size=hparams.batch_size,
                              drop_last=True,
                              collate_fn=collate_fn)

    return train_loader, valset, collate_fn


def load_checkpoint(checkpoint_path, G, D, G_optim, D_optim):
    assert os.path.isfile(checkpoint_path)
    print("Loading checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')

    G.load_state_dict(checkpoint_dict['G_state_dict'])
    D.load_state_dict(checkpoint_dict['D_state_dict'])
    G_optim.load_state_dict(checkpoint_dict['G_optimizer'])
    D_optim.load_state_dict(checkpoint_dict['D_optimizer'])

    iteration = checkpoint_dict['iteration']
    min_G_val_loss = checkpoint_dict['min_G_val_loss']

    print("Loaded checkpoint '{}' from iteration {}" .format(checkpoint_path, iteration))

    return G, D, G_optim, D_optim, iteration, min_G_val_loss

def load_checkpoint_model_only(checkpoint_path, G, D):
    assert os.path.isfile(checkpoint_path)
    print("Loading checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')

    G.load_state_dict(checkpoint_dict['G_state_dict'])
    D.load_state_dict(checkpoint_dict['D_state_dict'])

    print("Loaded checkpoint '{}'" .format(checkpoint_path))

    return G, D

def save_checkpoint(G, D, G_optim, D_optim, iteration, min_G_val_loss, filepath):
    print("Saving model and optimizer state at iteration {} to {}".format(iteration, filepath))
    torch.save({'iteration': iteration,
                'G_state_dict': G.state_dict(),
                'D_state_dict': D.state_dict(),
                'G_optimizer': G_optim.state_dict(),
                'D_optimizer': D_optim.state_dict(),
                'min_G_val_loss': min_G_val_loss}, filepath)


def validate(G, D, real_grid, fake_grid, criterion, valset, batch_size, collate_fn):
    G.eval()
    D.eval()
    with torch.no_grad():
        val_loader = DataLoader(valset,
                                shuffle=False, 
                                batch_size=batch_size,
                                drop_last=True,
                                collate_fn=collate_fn)

        G_avg_loss = 0.0
        D_avg_loss = 0.0
        i = 0
        for input, target in val_loader:
            G_loss, D_loss = criterion(G, D, input, target, real_grid, fake_grid)

            G_avg_loss += G_loss.item()
            D_avg_loss += D_loss.item()

            i += 1

        G_avg_loss /= (i + 1)
        D_avg_loss /= (i + 1)

    G.train()
    D.train()

    return G_avg_loss, D_avg_loss


def train(hparams, rank):
    if hparams.n_gpus > 1:
        init_distributed(hparams, rank)

    G = ResUNet(hparams).apply(weights_init).cuda()
    D = NPatchDiscriminators(hparams).apply(weights_init).cuda()

    criterion = GANLoss(hparams)

    G_optim = torch.optim.Adam(G.parameters(), lr=hparams.lr, betas=(hparams.beta1, hparams.beta2), eps=hparams.epsilon)
    D_optim = torch.optim.Adam(D.parameters(), lr=hparams.lr, betas=(hparams.beta1, hparams.beta2), eps=hparams.epsilon)

    lmbda = lambda epoch: hparams.lr_decay_factor
    G_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(G_optim, lr_lambda=lmbda)
    D_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(D_optim, lr_lambda=lmbda)

    train_loader, valset, collate_fn = prepare_dataloaders(hparams)

    # Load checkpoint if one exists
    min_G_val_loss = 1000
    G_val_loss = 0.0
    D_val_loss = 0.0
    iteration = 0
    epoch_offset = 0
    if hparams.checkpoint_path is not None:
        if hparams.load_model_state_dict_only == False:
            G, D, G_optim, D_optim, iteration, min_G_val_loss = load_checkpoint(hparams.checkpoint_path, G, D, G_optim, D_optim)

            iteration += 1  # next iteration is iteration + 1
            epoch_offset = max(0, int(iteration / len(train_loader)))
        else:
            G, D = load_checkpoint_model_only(hparams.checkpoint_path, G, D)

        G_val_loss, D_val_loss = validate(G, D, criterion, valset, hparams.batch_size, collate_fn)

    
    checkpoint_path = os.path.join(hparams.output_directory, "DUMMY")
    best_checkpoint_path = os.path.join(hparams.output_directory, "DUMMY")

    G.train()
    D.train()

    scaler = torch.cuda.amp.GradScaler(enabled=hparams.use_fp16)
    real_grid, fake_grid = build_D_tgt_grids(D, [512,512], hparams.batch_size)

    for epoch in range(epoch_offset, hparams.epochs):
        for input, target in train_loader:
            with torch.cuda.amp.autocast(enabled=hparams.use_fp16):
                G_loss, D_loss = criterion(G, D, input, target, real_grid, fake_grid)

            G_optim.zero_grad()
            scaler.scale(G_loss).backward()
            if hparams.clip_grad:
                scaler.unscale_(G_optim)
                torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=hparams.clip_grad_thresh)
            scaler.step(G_optim)
            scaler.update()

            D_optim.zero_grad()
            scaler.scale(D_loss).backward()
            if hparams.clip_grad:
                scaler.unscale_(D_optim)
                torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=hparams.clip_grad_thresh)
            scaler.step(D_optim)
            scaler.update()

            G_train_loss = G_loss.item()
            D_train_loss = D_loss.item()

            if(iteration % hparams.lr_decay_interval == 0):
                G_scheduler.step()
                D_scheduler.step()

            if (iteration % hparams.iters_per_checkpoint == 0):
                # G_val_loss, D_val_loss = validate(G, D, real_grid, fake_grid, criterion, valset, hparams.batch_size, collate_fn)

                if G_val_loss < min_G_val_loss and iteration > 0:
                    if(os.path.exists(best_checkpoint_path)):
                        os.remove(best_checkpoint_path)
                        
                    min_G_val_loss = G_val_loss
                    best_checkpoint_path = os.path.join(hparams.output_directory, "checkpoint_bestG_{}_{:.6f}".format(iteration, min_G_val_loss))
                    save_checkpoint(G, D, G_optim, D_optim, iteration, min_G_val_loss, best_checkpoint_path)
                            
                if iteration > 0:
                    if(os.path.exists(checkpoint_path)):
                        os.remove(checkpoint_path)

                    checkpoint_path = os.path.join(hparams.output_directory, "checkpoint_{}".format(iteration))
                    save_checkpoint(G, D, G_optim, D_optim, iteration, min_G_val_loss, checkpoint_path)
            
            print("{} {:.6f} {:.6f} {:.6f} {:.6f}".format(iteration, G_train_loss, D_train_loss, G_val_loss, D_val_loss))
            # with open('loss.txt', 'a') as f:
            #     f.write(str(iteration) + ' ' + str(G_train_loss) + ' ' + str(D_train_loss) + ' ' + str(G_val_loss) + ' ' + str(D_val_loss) + '\n')               

            iteration += 1

 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', type=int, default=0, required=False, help='rank of current gpu')
    args = parser.parse_args()

    hparams = HyperParams()
    train(hparams, args.rank)

if __name__ == '__main__':
    main()
    