import torch
from torch import nn, optim, utils
from torch.nn import functional as F
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.models import vgg as vgg
import numpy as np

class build_model(nn.Module):
    def __init__(self):                
        super().__init__()
        # network
        node_i = 784
        node_m = 512
        node_o = 10
        self.fm = nn.Linear(node_i, node_m)
        self.fo = nn.Linear(node_m, node_o)
        # loss function
        self.criterion = nn.MSELoss()
        # optimization function
        self.optimizer = optim.Adam(self.parameters())
        print ("model created")

    def forward(self, x):
        x = self.fm(x)
        # print ("middle layer calc done:", x)
        x = F.relu(x)
        # print ("middle layer activation done:", x)
        x = self.fo(x)
        # print ("out layer calc done:", x)
        x = F.relu(x)
        # print ("out layer activation done:", x)
        return x
    
def load_mnist():
    # For Jupyter, 2 is working but in shell ONLY 0 is acceptable why ??
    NUM_WORKERS = 0 
    
    trainset = datasets.MNIST(root='./data', train=True, download=False, transform=transforms.ToTensor())
    testset = datasets.MNIST(root='./data', train=False, download=False, transform=transforms.ToTensor())
    train_loader = utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=NUM_WORKERS) # for train, shuffle=True
    test_loader = utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=NUM_WORKERS) # for train, shuffle=True
    print ("loading mnist done")
    return train_loader, test_loader

def train(model, train_loader):
    # let model know it is for training
    model.train()

    total_correct = 0
    total_loss = 0
    total_data_len = 1
    accuracy = 0
    loss = 0

    # loop by minibatch
    for batch_imgs, batch_labels in train_loader:
        # convert 2d image data to 1d array
        batch_imgs = batch_imgs.reshape(-1, 28*28*1)
        # convert 10 elements array to one-hot array : only one answer data will be 1 and others becomes 0
        labels = torch.eye(10)[batch_labels]

        outputs = model(batch_imgs)
        model.optimizer.zero_grad()
        loss = model.criterion(outputs, labels)
        loss.backward()
        model.optimizer.step()

        # ミニバッチごとの正答率と損失を求める
        _, pred_labels = torch.max(outputs, axis=1)  # outputsから必要な情報(予測したラベル)のみを取り出す。
        batch_size = len(batch_labels)  # バッチサイズの確認
        for i in range(batch_size):  # データ一つずつループ,ミニバッチの中身出しきるまで
            total_data_len += 1  # 全データ数を集計
            if pred_labels[i] == batch_labels[i]:
                total_correct += 1 # 正解のデータ数を集計
        total_loss += loss.item()  # 全損失の合計

    # 今回のエポックの正答率と損失を求める
    accuracy = total_correct/total_data_len*100  # 予測精度の算出
    loss = total_loss/total_data_len  # 損失の平均の算出
    return accuracy, loss

def run_train():
    model = build_model()
    train_loader, test_loader = load_mnist()
    acc, loss = train(model, train_loader)
    print(f'正答率: {acc}, 損失: {loss}')

run_train()

# MLP(Multi Layer Perceptron)を構成する全結合層(Fully Connected Layer)は
# nn.Linear()クラスで用いる。
# 引数はin_featuresとout_featuresで入力と出力のノード数を指定する。
# 伝播させる時は__call__()メソッドの引数にデータを渡す。

def check_versions():
    print ("torch version:", torch.__version__)
    print ("torchivision version:", torchvision.__version__)
    print ("numpy version:", np.__version__)

# forward propagation : 純伝播
def fppgt():
    fc = nn.Linear(4,2)
    x = torch.Tensor([[1,2,3,4]])
    print (x)
    x = fc(x)   
    print (x)

# fppgt()

def actvf():
    x = torch.Tensor([[-0.9, -0.1, 0, 0.4, 0.5, 0.99, 1.2]])
    x = F.relu(x)
    print (x)

# actvf()

# 平均２乗法 (MSE)
def test_mse():
    criterion = nn.MSELoss()
    x = torch.Tensor([[1,1,1,1]])   # 純伝播の結果
    y = torch.Tensor([[0,2,4,6]])  # 正解ラベル
    loss = criterion(x, y)          # 損失値

    # 均２乗法
    # (0 - 1)^2 + (1 - (-1))^2 + (2 - 0)^2 / 3
    print (loss)

# test_mse()

def test_adam():
    model = vgg.vgg11()
    optimizer = optim.Adam(model.parameters())
    print (optimizer)

    # from results:
    # lr = learning rate (0.001 is commonly best value)
    #

# test_adam()

