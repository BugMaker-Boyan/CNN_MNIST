from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from torch.utils.data import DataLoader

BATCH_SIZE = 128
TEST_BATCH_SIZE = 512


def get_dataLoader(train=True):
    batch_size = BATCH_SIZE if train else TEST_BATCH_SIZE
    train_data = MNIST(root='.',
                       train=train,
                       transform=Compose([
                           ToTensor(),
                           Normalize(mean=(0.1307,), std=(0.3081,)),
                       ]),
                       download=True)
    data_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=train)
    return data_loader


# loader = get_dataLoader(True, BATCH_SIZE)
# for idx, (x, y) in enumerate(loader):
#     print(x)
#     print(x.size())
#     print(y)
#     print(y.size())
#
#     break
