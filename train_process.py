import torch
import torch.nn as nn
import photontorch as pt
from torchvision import datasets
import torchvision.transforms as transforms
from deep_learning_utils import DLModule
from ring_model import Model

wavelength_list = [1.3e-6, 1.35e-6, 1.4e-6, 1.45e-6, 1.5e-6, 1.55e-6, 1.6e-6, 1.65e-6]
env = pt.Environment(t_start = 0, t_end = 1e-11, dt = 1e-12, wl = wavelength_list, grad=True, freqdomain=True)
pt.set_environment(env)

# 搭建网络
model = Model()
# 超参数
train_batch = 3000
test_batch = 1000
loss_fn = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=0.01)
# 数据集
trainset = datasets.MNIST(train=True, transform=transforms.ToTensor(), download=True)
testset = datasets.MNIST(train=False, transform=transforms.ToTensor(), download=True)

# 训练测试模块
dl = DLModule(model=model, loss_fn=loss_fn, optim=optim, train_set=trainset, 
              test_set=testset, train_batch=train_batch, test_batch=test_batch)
best_accuracy = 0
for epoch in range(100):
    print(f"——————————————Epoch:{epoch}——————————————")
    dl.train_mode()
    dl.test_mode()
    dl.epoch += 1
    if dl.accuracy > best_accuracy :
        best_accuracy = dl.accuracy
        torch.save(model.state_dict(), "best_model.pth")