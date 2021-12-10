# This part is borrowed from https://github.com/huawei-noah/Data-Efficient-Model-Compression
import torch.nn.utils.weight_norm as weightNorm
import torch.nn as nn


def init_weights(m):     
    classname = m.__class__.__name__ 
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1: 
        nn.init.kaiming_uniform_(m.weight) 
        nn.init.zeros_(m.bias) 
    elif classname.find('BatchNorm') != -1: 
        nn.init.normal_(m.weight, 1.0, 0.02)  
        nn.init.zeros_(m.bias) 
    elif classname.find('Linear') != -1: 
        nn.init.xavier_normal_(m.weight) 
        nn.init.zeros_(m.bias)   

 

class LeNet5(nn.Module):

    def __init__(self):
        super(LeNet5, self).__init__()

        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)
        self.maxpool1 = nn.MaxPool2d(2) 
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5)
        self.dropout = nn.Dropout2d(p=0.5)
        self.maxpool2 = nn.MaxPool2d(2)
        self.relu2 = nn.ReLU()



        self.fc1 = nn.Linear(800, 256)
        
        self.bn = nn.BatchNorm1d(256, affine=True)
        self.dropout2 = nn.Dropout2d(p=0.5)

        self.fc2 = weightNorm(nn.Linear(256, 10), name="weight")  


    def forward(self, img, out_feature=False):
        output = self.conv1(img)
        output = self.maxpool1(output)
        output = self.relu1(output)
        output = self.conv2(output)
        output = self.dropout(output)
        output = self.maxpool2(output)
        output = self.relu2(output)

        feature = output.view(-1, 800)

        output = self.fc1(feature)
        output = self.bn(output)
        output = self.dropout2(output)
        output = self.fc2(output)

        if out_feature == False:
            return output
        else:
            return output,feature


    

class LeNet5Half(nn.Module):

    def __init__(self):
        super(LeNet5Half, self).__init__()

        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.maxpool1 = nn.MaxPool2d(2) 
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(10, 25, kernel_size=5)
        self.dropout = nn.Dropout2d(p=0.5)
        self.maxpool2 = nn.MaxPool2d(2)
        self.relu2 = nn.ReLU()



        self.fc1 = nn.Linear(400, 256)
        
        self.bn = nn.BatchNorm1d(256, affine=True)
        self.dropout2 = nn.Dropout2d(p=0.5)

        self.fc2 = weightNorm(nn.Linear(256, 10), name="weight")  


    def forward(self, img, out_feature=False):
        output = self.conv1(img)
        output = self.maxpool1(output)
        output = self.relu1(output)
        output = self.conv2(output)
        output = self.dropout(output)
        output = self.maxpool2(output)
        output = self.relu2(output)

        feature = output.view(-1, 400)

        output = self.fc1(feature)
        output = self.bn(output)
        output = self.dropout2(output)
        output = self.fc2(output)

        if out_feature == False:
            return output
        else:
            return output,feature



class LeNetBase(nn.Module):
    def __init__(self):
        super(LeNetBase, self).__init__()
        self.conv_params = nn.Sequential(
                nn.Conv2d(1, 20, kernel_size=5),
                nn.MaxPool2d(2),
                nn.ReLU(),
                nn.Conv2d(20, 50, kernel_size=5),
                nn.Dropout2d(p=0.5),
                nn.MaxPool2d(2),
                nn.ReLU(),
                )
        self.in_features = 50*4*4

    def forward(self, x):
        x = self.conv_params(x)
        #x = x.view(x.size(0), -1)
        x = x.view(10)
        return x
