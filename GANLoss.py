import torch
import torch.nn as nn


def get_grid(input, is_real=True):
    if is_real:
        grid = torch.cuda.FloatTensor(input.shape).fill_(1.0)

    elif not is_real:
        grid = torch.cuda.FloatTensor(input.shape).fill_(0.0)

    return grid
    

class GANLoss(object):
    def __init__(self, hparams):
        self.n_D = hparams.n_D
        self.lambda_FM = hparams.lambda_FM

        self.criterion = nn.MSELoss()
        self.FMcriterion = nn.L1Loss()

    def __call__(self, G, D, input, target, real_grid, fake_grid):
        loss_D = 0
        loss_G = 0
        loss_G_FM = 0

        fake = G(input)

        real_features = D(torch.cat((input, target), dim=1))
        fake_features = D(torch.cat((input, fake.detach()), dim=1))

        for i in range(self.n_D):
            loss_D += (self.criterion(real_features[i][-1], real_grid[i]) +
                       self.criterion(fake_features[i][-1], fake_grid[i])) * 0.5

        fake_features = D(torch.cat((input, fake), dim=1))

        for i in range(self.n_D):
            loss_G += self.criterion(fake_features[i][-1], real_grid[i])
            
            for j in range(len(fake_features[0])):
                loss_G_FM += self.FMcriterion(fake_features[i][j], real_features[i][j].detach())
                
            loss_G += loss_G_FM * (1.0 / self.n_D) * self.lambda_FM

        return loss_G, loss_D