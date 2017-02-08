from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import torch
import torch.nn as nn

class SumLoss(nn.Module):
    #def __init__(self, isize, nz, nc, ndf, ngpu):
    #    super(SumLoss, self).__init__()
    #    self.ngpu = ngpu

    #    main = nn.Sequential(
    #        # Z goes into a linear of size: ndf
    #        nn.Linear(nc * isize * isize, ndf),
    #        nn.ReLU(True),
    #        nn.Linear(ndf, ndf),
    #        nn.ReLU(True),
    #        nn.Linear(ndf, ndf),
    #        nn.ReLU(True),
    #        nn.Linear(ndf, 1),
    #    )
    #    self.main = main
    #    self.nc = nc
    #    self.isize = isize
    #    self.nz = nz

    def __init__(self, sign=1):
        super(SumLoss, self).__init__()
        self.sign = sign

    def forward(self, input, target=0):
        output = torch.mul(input, self.sign)
        output = output.view(input.size(0),
                           input.size(1) * input.size(2) * input.size(3))
        output = output.mean(0)
        return output.view(1)
