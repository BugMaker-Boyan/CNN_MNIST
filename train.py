from model import CNN
import torch
from dataset import get_dataLoader
from torch.optim import Adam
import os
import torch.nn.functional as F
import numpy as np

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = CNN().to(DEVICE)
optimizer = Adam(model.parameters(), lr=0.01)
if os.path.exists('./model/model.pkl'):
    model.load_state_dict(torch.load('./model/model.pkl'))
    optimizer.load_state_dict(torch.load('./model/optimizer.pkl'))


def train(epoch):
    for idx, (x, target) in enumerate(get_dataLoader(True)):
        x = x.to(DEVICE)
        target = target.to(DEVICE)
        optimizer.zero_grad()
        output = model(x)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if idx % 100 == 0:
            torch.save(model.state_dict(), 'model/model.pkl')
            torch.save(optimizer.state_dict(), 'model/optimizer.pkl')
            print("epoch:{}, idx:{}, loss:{}".format(epoch, idx, loss))


def test():
    model.eval()
    loss_list = []
    acc_list = []
    for idx, (x, target) in enumerate(get_dataLoader(False)):
        with torch.no_grad():
            x = x.to(DEVICE)
            target = target.to(DEVICE)
            output = model(x)
            loss = F.nll_loss(output, target)
            loss_list.append(loss.cpu())
            predict = output.data.max(dim=-1)[1]
            acc = predict.eq(target).float().mean()
            acc_list.append(acc.cpu())
    print('Average accuracy rate: {}, Average loss: {}.'.format(np.mean(acc_list), np.mean(loss_list)))


if __name__ == '__main__':
    for i in range(30):
        train(i)
    test()