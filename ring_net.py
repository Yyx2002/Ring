import torch
import photontorch as pt
import numpy as np

class Ring(pt.Network):
    def __init__(self, wavelength=1.55e-6, phase=0, neff=3.4, loss=0):
        super(Ring, self).__init__()
        self.neff = neff
        self.wavelength = wavelength
        self.wg_length = wavelength / (self.neff * 2)# 单个直波导的长度
        self.loss = loss
        self.phase = phase
        self.add_component("dc1", pt.DirectionalCoupler(coupling=0.05, trainable=False))
        self.add_component("dc2", pt.DirectionalCoupler(coupling=0.05, trainable=False))
        self.add_component("wg1", pt.Waveguide(length = self.wg_length, loss=self.loss, phase = 0, trainable = False, neff=self.neff))
        self.add_component("wg2", pt.Waveguide(length = self.wg_length, loss=self.loss, phase = self.phase, trainable = True, neff=self.neff)) 
        self.add_component("source", pt.Source())
        self.add_component("detector1", pt.Detector())
        self.add_component("detector2", pt.Detector())
        self.add_component("term", pt.Term())
        self.link(0, "0:dc1:3", "0:wg1:1", "0:dc2:1", "1:wg2:0", "2:dc1:1", 1)
        self.link(3, "3:dc2:2", 2)

class RingLink(pt.Network):
    def __init__(self, wavelength, phase, neff):
        # wavelength 表示构建微环的谐振波长
        super(RingLink, self).__init__()
        self.add_component("source", pt.Source())
        self.add_component("detector1", pt.Detector())
        self.add_component("detector2", pt.Detector())
        self.add_component("term", pt.Term())
        self.ring = Ring(wavelength=wavelength, phase=phase, neff=neff, loss=0)

        self.link("source:0", "0:ring:1", "0:detector1")
        self.link("detector2:0", "3:ring:2", "0:term")

class Ring_MN_Array(pt.Network):
    def __init__(self, M, N, mode, wavelength_list, phase, neff, loss):
        super(Ring_MN_Array, self).__init__()
        # 构建相应个数M✖️N的微环
        for j in range(N):
            for wl in wavelength_list:
                for i in range(M):
                    self.add_component(f"ring{i}_{j}", Ring(wavelength=wl))
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

class RingNet(pt.Network):
    def __init__(self, M, N, mode=1, wavelength_list=1e-6 * np.linspace(1.3, 1.65, 8), phase=0, neff=3.4, loss=0):
        super(RingNet, self).__init__()
        self.wavelength_list = wavelength_list
        self.ring = Ring_MN_Array(M, N, mode, self.wavelength_list, phase, neff, loss)
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
    wavelength_list = [1.3e-6, 1.35e-6, 1.4e-6, 1.45e-6, 1.5e-6, 1.55e-6, 1.6e-6, 1.65e-6]
    net = RingNet(10, 8, mode=1, wavelength_list=wavelength_list, phase=0, neff=3.4, loss=0)