import torch
from torch.utils.data import  Dataset
import random
from matplotlib import pyplot as plt
import photontorch as pt
import numpy as np
from numpy import arange

class SingleRingStructure(pt.Network):
    def __init__(self, wavelength):
        super(SingleRingStructure, self).__init__()
        neff = 2.34
        ring_length = 425.8734943010671 * wavelength / neff
        self.add_component("dc1", pt.RealisticDirectionalCoupler(k0=0.8, wl0 = wavelength))
        self.add_component("dc2", pt.RealisticDirectionalCoupler(k0=0.8, wl0 = wavelength))
        self.add_component("wg", pt.Waveguide(length = ring_length, loss=1500, wl0 = wavelength, trainable = False))
        self.add_component("mzi", pt.Mzi(wl0 = wavelength))
        # Term 连接MZI多余端口
        self.add_component("term1", pt.Term())
        self.add_component("term2", pt.Term())        
        self.link(0, "0:dc1:3", "0:wg:1", "0:dc2:1", "0:mzi:1", "2:dc1:1", 1)
        self.link("term1:0", "2:mzi:3", "0:term2")
        self.link(3, "3:dc2:2", 2)

class Ring_MN_Array(pt.Network):
    def __init__(self, M, N, mode, wavelength_list):
        super(Ring_MN_Array, self).__init__()
        # 构建相应个数M✖️N的微环
        for j in range(N):
            for wl in wavelength_list:
                for i in range(M):
                    self.add_component(f"ring{i}_{j}", SingleRingStructure(wl))
        if mode == 0 : # 仅探测drop端 
            for i in range(M):
                self.add_component(f"term{i}_1", pt.Term())
                self.add_component(f"term{i}_2", pt.Term())
                self.link(i, f"0:ring{i}_0:3", M+i)
                self.link(f"term{i}_1:0", f"1:ring{i}_{N-1}:2", f"0:term{i}_2")
                for j in range(N-1):
                    self.link(f"ring{i}_{j}:1", f"0:ring{i}_{j+1}")
                    self.link(f"ring{i}_{j}:2", f"3:ring{i}_{j+1}")
        if mode == 1 : # 探测drop&through端 
            for i in range(M):
                self.add_component(f"term{i}", pt.Term())
                self.link(i, f"0:ring{i}_0:3", M+i)
                self.link(2*M+i, f"1:ring{i}_{N-1}:2", f"0:term{i}")
                for j in range(N-1):
                    self.link(f"ring{i}_{j}:1", f"0:ring{i}_{j+1}")
                    self.link(f"ring{i}_{j}:2", f"3:ring{i}_{j+1}")


class FullNet(pt.Network):
    def __init__(self, M, N, mode):
        super(FullNet, self).__init__()
        wavelength_list = 1e-6*np.linspace(1.55, 1.6, N)
        self.ring = Ring_MN_Array(M, N, mode, wavelength_list)
        # 添加光源
        for i in range(M):
            self.add_component(f"source{i}",pt.Source())
            self.link(f"source{i}:0", f"{i}:ring")

        if mode == 0 :
            for i in range(M):
                self.add_component(f"detector{i}", pt.Detector())
                self.link(f"ring:{M+i}", f"0:detector{i}")
        if mode == 1 :
            for i in range(2*M):
                self.add_component(f"detector{i}", pt.Detector())
                self.link(f"ring:{M+i}", f"0:detector{i}")

if __name__ == "__main__":
    env = pt.Environment(t0=0, t1=1e-11, dt=1e-12, wl=1e-6*np.linspace(1.5, 1.6, 3), grad=True)
    pt.set_environment(env)
    net = FullNet(M=4, N=3, mode=0)
    out = net(source = 1)
    net.plot(out)
    plt.show()