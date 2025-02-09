try:
    import torch
    import torchvision
    import numpy as np
    from torch import nn
    import torch.nn.functional as F
    print ("loaded modules")
    print ("torch version:", torch.__version__)
    print ("torchivision version:", torchvision.__version__)
    print ("numpy version:", np.__version__)
except:
    print ("fail to load modules")
    exit ()

def check_types():
    nums1 = np.zeros((2,3))
    print (nums1)
    print ("np:", nums1.shape, nums1.dtype)

    nums2 = torch.zeros((2,3))
    print (nums2)
    print ("torch:", nums2.shape, nums2.dtype)

    # pass type
    nums3 = torch.ones((2,3), dtype=torch.int64)
    print (nums3)
    print ("torch:", nums3.shape, nums3.dtype)

# check_types()

# Sharing the memory of origin 
# - changing original variable impacting to others converted from origin
# - default data type will be copied to the new varibale
# -----------------------------------------
def conv_types():
    tens = torch.zeros(2,3)
    print ("created tensor.")
    print ("tens:", tens, tens.dtype)
    nump = tens.numpy()
    print ("nump:", nump, nump.dtype)
    tens1 = torch.from_numpy(nump)
    print ("tensor converted from numpy:")
    print ("tens1:", tens1, tens1.dtype)

    tens.add_(1)
    print ("changed to only original tensor")
    print ("tens:", tens)
    print ("nump:", nump)
    print ("tens1:", tens1)

# conv_types()

def use_gpu():
    if torch.cuda.is_available():
        gpu = torch.device("cuda")
        cpu = torch.device("cpu")
        data_gpu = torch.zeros((2,2), device=gpu)
        data_cpu = data_gpu.to(cpu)
        data_gpu_new = data_cpu.to(gpu)
        print (data_cpu)
        print (data_gpu_new)
    else:
        print ("not available to use GPU")

# use_gpu()

def calc_tensor():
    nums1 = torch.zeros(2,3)
    nums2 = torch.ones(2,3)
    print (nums1*2 + nums2*3)

# calc_tensor()

def use_gradient():
    x = torch.tensor(1.0, requires_grad=True)
    a, b = 3, 5

    # 純伝播
    y = a*x + b
    print ("結果:", y)

    # 逆伝播
    y.backward()
    print ("勾配:", x.grad)

    # 偏微分
    v, w = torch.tensor(3.0, requires_grad=True), torch.tensor(39.0, requires_grad=True)
    a, b, c = 4, 6, 1
    z = a*v + b*w + c
    z.backward()
    print ("vで偏微分した時の勾配:", v.grad)
    print ("wで偏微分した時の勾配:", w.grad)

# use_gradient()

def conv_shape():
    nums1 = torch.rand((4,3))
    print (nums1)
    nums2 = nums1.transpose(1,0)
    print (nums2)
    nums3 = torch.t(nums2)
    print (nums3)

    ones1 = torch.ones(16)
    print (ones1)
    ones2 = ones1.reshape(2,-1) # -1 for automatic
    print (ones2)
    ones3 = ones2.reshape(4,-1)
    print (ones3)

    x = torch.tensor([[1,1,1], [1,2,2], [2,2,3], [3,3,3]])
    print (x)
    y = x.reshape(1,-1)
    print (y)
    z = y.reshape(3,-1)
    print (z)

# conv_shape()

def under_bar_function():
    nums = torch.tensor([[0,1], [2,3]])
    print (nums)
    nums.zero_() # set zero to all elements
    print (nums)

# under_bar_function()

def run_activation_f():
    x = torch.Tensor(([-2, 999, 0, 0.3, 1]))
    print (x)

    print ('relu:', F.relu(x))
    print ('softmax:', F.softmax(x, dim=0)) #dim : dimension
    print ('sigmoid:', torch.sigmoid(x))

# run_activation_f()

def run_loss_f():
    # model output : sum = 1
    x = torch.tensor([[0.2, 0.1, 0.7]])

    # MSE label for answer : one-hot vector
    ans_mse = torch.tensor([[0, 1, 0]])
    print ('MCE:', nn.MSELoss(reduction='mean')(x, ans_mse))

    # cross entropy label for answer : scalar
    ans_cel = torch.tensor([1])
    print ('CrossEntropy:', nn.CrossEntropyLoss()(x, ans_cel))

run_loss_f()

