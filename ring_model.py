import torch.nn as nn
import torch
from ring_net import RingNet

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layer_flat = torch.nn.Flatten()# output: (batch, 1 * 28 * 28)
        self.layer_el = torch.nn.Linear(1 * 28 * 28, 4)# output: (batch, 64)
        self.layer_relu = torch.nn.GELU()
        self.layer_ol = RingNet(10, 4, mode="t", wavelength_list=[1.53e-6, 1.54e-6, 1.55e-6, 1.56e-6])

    def forward(self, x):
        x = self.layer_flat(x)
        x = self.layer_el(x)
        x = self.layer_relu(x)
        """
        将8个数据用8个波长承载，复用之后输入从同一个光源端口输入
        """
        x = torch.chunk(x, 4, dim = -1)
        x = torch.stack(x, dim = 0)
        x = torch.squeeze(x, dim = -1)
        x = torch.stack([x] * 10, dim = 0).rename('s', 'w', 'b')
        x = self.layer_ol(source = x)[-1, :, :, :]
        # print(torch.max(x, 0, keepdim=True))
        # print(torch.max(x, 1, keepdim=True))
        # print(torch.max(x, 2, keepdim=True))
        x = torch.sum(x, dim = 0) # 在波长维度求和
        x = torch.transpose(x, 0, 1)
        # x = torch.chunk(x, 2, dim = -1)
        # x = x[0] - x[1] # 进行差分探测
        return x