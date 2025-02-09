import torch
from torch import nn, optim, utils
from torch.nn import functional as F

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.models import vgg as vgg
import numpy as np

def check_versions():
    print ("torch version:", torch.__version__)
    print ("torchivision version:", torchvision.__version__)
    print ("numpy version:", np.__version__)

# check_versions()

INPUT_NODE_CNT = 784
MIDDLE_NODE_CNT = 512
OUTPUT_NODE_CNT = 10

# ------------------------------------------------------------------
# MLP(Multi Layer Perceptron)を構成する全結合層(Fully Connected Layer)は
# nn.Linear()クラスで用いる。
# 引数はin_featuresとout_featuresで入力と出力のノード数を指定する。
# 伝播させる時は__call__()メソッドの引数にデータを渡す。
# ------------------------------------------------------------------
class build_model(nn.Module):
    def __init__(self):                
        super().__init__()
        # network
        self.fm = nn.Linear(INPUT_NODE_CNT, MIDDLE_NODE_CNT)
        self.fo = nn.Linear(MIDDLE_NODE_CNT, OUTPUT_NODE_CNT)

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
    PERFORM_DOWNLOAD = False

    # download mnist dataset
    trainset = datasets.MNIST(root='./data', train=True, download=PERFORM_DOWNLOAD, transform=transforms.ToTensor())
    testset = datasets.MNIST(root='./data', train=False, download=PERFORM_DOWNLOAD, transform=transforms.ToTensor())
    # load dataset
    train_loader = utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=NUM_WORKERS) # for train, shuffle=True
    test_loader = utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=NUM_WORKERS) # for train, shuffle=True
    print ("loading mnist done")
    return train_loader, test_loader

def train(model, data_loader):
    # let model know it is for training
    model.train()

    # initialize info variables
    total_data_len = 1
    total_correct = 0
    total_loss = 0
    accuracy = 0
    loss = 0

    # loop by minibatch
    print ("learning ...\n")
    for batch_imgs, batch_labels in data_loader:
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

def run_train(model, train_loader):
    acc, loss = train(model, train_loader)
    print(f'正答率: {acc}, 損失: {loss}')
    print ()

def test(model, data_loader):
    # let model know it is for testing
    model.eval()

    # initialize info variables
    total_data_len = 0
    total_correct = 0

    # loop by minibatch
    for batch_imgs, batch_labels in data_loader:
        outputs = model(batch_imgs.reshape(-1, 28*28*1))
        # no need to get one-hot array due to no need to runn loss func

        # gather predictions by mini-batch
        _, pred_labels = torch.max(outputs, axis=1)
        batch_size = len(pred_labels)
        for i in range(batch_size):
            total_data_len += 1
            if (batch_labels[i] == pred_labels[i]):
                total_correct += 1

    return total_data_len, total_correct

def run_test(model, test_loader):
    total, correct = test(model, test_loader)
    accuracy = 100.0 * correct / total 
    print (f'evaluation result: correct={correct}, total={total}: accuracy={accuracy}%')
    print ()

# -------------------------------
# train and test
# -------------------------------
# model = build_model()
# train_loader, test_loader = load_mnist()
# run_train(model, train_loader)
# run_test(model, test_loader)
# -------------------------------

# forward propagation : 純伝播
def test_forward():
    fc = nn.Linear(4,2)
    x = torch.Tensor([[1,2,3,4]])
    print (x)
    x = fc(x)   
    print (x)

# test_forward()

def test_activation_f():
    x = torch.Tensor([[-0.9, -0.1, 0, 0.4, 0.5, 0.99, 1.2]])
    x = F.relu(x)
    print (x)

# test_activation_f()

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

def test_optimization():
    model = vgg.vgg11()

    sgd = optim.SGD(model.parameters(), lr=0.01)
    momentum = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    adam = optim.Adam(model.parameters())

    print (sgd)
    print (momentum)
    print (adam)

    # from the result:
    # lr = learning rate (0.001 is commonly best value)
    #

# test_optimization()

# 畳み込み層
def test_cnn():
    INPUT_CHANNEL = 3
    OUTPUT_CHANNEL = 3
    KERNEL_SIZE = 3
    conv = nn.Conv2d(INPUT_CHANNEL, OUTPUT_CHANNEL, KERNEL_SIZE)

    # １枚の画像で、３色のRGBの、28*28のサイズの画像を使用
    IMAGES_NUMBER = 1
    RGB_DIMENSION = 3
    IMAGE_SIZE_H = 28
    IMAGE_SIZE_W = 28
    x = torch.Tensor(torch.rand(IMAGES_NUMBER, RGB_DIMENSION, IMAGE_SIZE_H, IMAGE_SIZE_W))
    x = conv(x)
    print (x.shape)

# test_cnn()

# ------------------------------------------
# using existing model
# ------------------------------------------
class model_vgg(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = torchvision.models.vgg11()
        # Output of vgg is 1000, so if want to make final ouput as 10, needs to use 全結合層
        self.fc = nn.Linear(1000, 10)

    def forward(self, x):
        x = self.vgg(x)
        x = self.fc(x)
        return x

class model_resnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = torchvision.models.resnet18()
        # Output of resnet is 1000, so if want to make final ouput as 10, needs to use 全結合層
        self.fc = nn.Linear(1000, 10)

    def forward(self, x):
        x = self.resnet(x)
        x = self.fc(x)
        return x

def test_existing_models():
    print ()
    vgg = model_vgg()
    print(vgg)
    print ()

    resnet = model_resnet()
    print(resnet)
    print ()

test_existing_models()


