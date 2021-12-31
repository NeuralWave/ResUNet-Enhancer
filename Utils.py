import torch
import torch.nn as nn

def get_grid_sizes(D, in_img_sz):
    input_dummy = torch.cuda.FloatTensor(1,1,in_img_sz[0],in_img_sz[1]).fill_(0.0)

    D.eval()
    with torch.no_grad():
        outputs = D(torch.cat((input_dummy, input_dummy), dim=1))

    grid_sizes = []
    for output in outputs:
        grid_sizes.append(output[-1].squeeze().size())

    D.train()
    return grid_sizes


def build_D_tgt_grids(D, in_img_sz, batch_size):
    real_grid = []
    fake_grid = []

    grid_sizes = get_grid_sizes(D, in_img_sz)

    for grid_size in grid_sizes:
        real_grid.append(torch.cuda.FloatTensor(batch_size, 1, grid_size[0], grid_size[1]).fill_(1.0))
        fake_grid.append(torch.cuda.FloatTensor(batch_size, 1, grid_size[0], grid_size[1]).fill_(0.0))

    return real_grid, fake_grid


def weights_init(module):
    if isinstance(module, nn.Conv2d):
        module.weight.detach().normal_(0.0, 0.02)

    elif isinstance(module, nn.BatchNorm2d):
        module.weight.detach().normal_(1.0, 0.02)
        module.bias.detach().fill_(0.0)