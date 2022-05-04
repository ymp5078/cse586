import torch
from torchvision import transforms, models
from torchsummary import summary

class vgg16(torch.nn.Module):
    def __init__(self,num_classes,input_channels=3):
        super(vgg16, self).__init__()
        self.model = models.vgg16_bn()
        if input_channels != 3:
            self.model.features[0] = torch.nn.Conv2d(in_channels=input_channels, out_channels=64, kernel_size=(3,3), stride=1, padding=1)
        last_layer_size = self.model.classifier[6].in_features
        self.model.classifier[6] = torch.nn.Linear(last_layer_size,num_classes)
        # self.fc1 = torch.nn.Linear(1000, 256)
        # self.activation = torch.nn.ReLU()
        # self.fc2 = torch.nn.Linear(256,num_class)
    
    def forward(self, x):
        x = self.model(x)
        # x = self.encoder(x)
        # x = self.fc1(x)
        # x = self.activation(x)
        # x = self.fc2(x)
        return x
    def get_target_layers(self):
        # get the layers that we want to generate GradCAM with
        return [self.model.features[19],
                self.model.features[22],
                self.model.features[26],
                self.model.features[29],
                self.model.features[32],
                self.model.features[36],
                self.model.features[39],
                self.model.features[42]]

class ResNet50(torch.nn.Module):
    def __init__(self,num_classes,finetune=False):
        super(ResNet50, self).__init__()
        self.model = models.resnet50()
        last_layer_size = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(last_layer_size,num_classes)
        if finetune:
            for name, param in self.model.named_parameters():
                if not ('fc' in name):
                    param.requires_grad = False
        # self.fc1 = torch.nn.Linear(1000, 256)
        # self.activation = torch.nn.ReLU()
        # self.fc2 = torch.nn.Linear(256,num_class)
    
    def forward(self, x):
        with torch.cuda.amp.autocast():
            x = self.model(x)
        # x = self.encoder(x)
        # x = self.fc1(x)
        # x = self.activation(x)
        # x = self.fc2(x)
        return x
    def get_target_layers(self):
        # get the layers that we want to generate GradCAM with
        return [self.model.layer4[-1]]

class ResNet18(torch.nn.Module):
    def __init__(self,num_classes,finetune=False):
        super(ResNet18, self).__init__()
        self.model = models.resnet18()
        last_layer_size = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(last_layer_size,num_classes)
        if finetune:
            for name, param in self.model.named_parameters():
                if not ('fc' in name):
                    param.requires_grad = False
        # self.fc1 = torch.nn.Linear(1000, 256)
        # self.activation = torch.nn.ReLU()
        # self.fc2 = torch.nn.Linear(256,num_class)
    
    def forward(self, x):
        # with torch.cuda.amp.autocast():
        x = self.model(x)
        # x = self.encoder(x)
        # x = self.fc1(x)
        # x = self.activation(x)
        # x = self.fc2(x)
        return x
    def get_target_layers(self):
        # get the layers that we want to generate GradCAM with
        return [self.model.layer4[-1]]

class SimpleCNN(torch.nn.Module):
    def __init__(self,num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, 3, 1)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = torch.nn.Dropout(0.25)
        self.dropout2 = torch.nn.Dropout(0.5)
        self.fc1 = torch.nn.Linear(9216, 128)
        self.fc2 = torch.nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        x = self.conv2(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        # output = torch.nn.functional.log_softmax(x, dim=1)
        return x
    def get_target_layers(self):
        # get the layers that we want to generate GradCAM with
        return [self.conv2]

MODELS = {
    'vgg16':vgg16,
    'ResNet50':ResNet50,
    'ResNet18':ResNet18,
    'SimpleCNN':SimpleCNN
}

def get_model(model_name):
    return MODELS[model_name]

if __name__=='__main__':
    device = torch.device('cpu')
    model = vgg16(10).cuda()#.to(device)
    print(model)
    # for name, param in model.named_parameters():print(name, param)

    summary(model,(3, 224, 224))