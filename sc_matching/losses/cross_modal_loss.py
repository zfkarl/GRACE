import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class CrossModalLoss(nn.Module):

    def __init__(self,temp = 0.22, modal_num =2):
        super(CrossModalLoss, self).__init__()
        self.temp = temp
        self.modal_num = modal_num

    def forward(self, x):
        batch_size = x.size(0)
        x = F.normalize(x, p=2, dim=1)
        
        batch_size = x.shape[0] // self.modal_num
        sim = x.mm(x.t())
        sim = (sim / self.temp).exp()
        sim = (sim - sim.diag().diag()) 

        sim_sum1 = sum([sim[:, v * batch_size: (v + 1) * batch_size] for v in range(self.modal_num)])

        diag1 = torch.cat([sim_sum1[v * batch_size: (v + 1) * batch_size].diag() for v in range(self.modal_num)])
        loss1 = -(diag1 / sim.sum(1)).log().mean()

        sim_sum2 = sum([sim[v * batch_size: (v + 1) * batch_size] for v in range(self.modal_num)])
        diag2 = torch.cat([sim_sum2[:, v * batch_size: (v + 1) * batch_size].diag() for v in range(self.modal_num)])
        loss2 = -(diag2 / sim.sum(1)).log().mean()
        return loss1 + loss2