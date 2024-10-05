import torch.nn as nn
import torch
from ring_net import RingNet


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layer_flat = torch.nn.Flatten()# output: (batch, 1 * 28 * 28)
        self.layer_el = torch.nn.Linear(1 * 28 * 28, 8)# output: (batch, 64)
        self.layer_relu = torch.nn.ReLU()
        self.layer_ol = RingNet(10, 8, mode=1, wavelength_list=[1.3e-6, 1.35e-6, 1.4e-6, 1.45e-6, 1.5e-6, 1.55e-6, 1.6e-6, 1.65e-6])
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        x = self.layer_flat(x)
        x = self.layer_el(x)
        x = self.layer_relu(x)
        """
        将8个数据用8个波长承载，复用之后输入从同一个光源端口输入
        """
        x = torch.chunk(x, 8, dim = -1)
        x = torch.stack(x, dim = 0)
        x = torch.squeeze(x, dim = -1)
        x = torch.stack([x] * 10, dim = 0).rename('s', 'w', 'b')
        x = self.layer_ol(source = x)[-1, :, :, :]
        x = torch.permute(x, (-1, 0, 1))
        x = torch.sum(x, dim = 1)
        x = torch.chunk(x, 2, dim = 1)
        x = x[0] - x[1] # 进行查分探测
        x = self.softmax(x)
        return x