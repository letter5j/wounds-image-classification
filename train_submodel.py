
from train import start_train
from evaluate import eval_model


import torch
import torch.nn as nn
from torch.autograd import Variable
import pretrainedmodels
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
def freeze_layer(layer):
    for param in layer.parameters():
        param.requires_grad = False
def build_model(model_name):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load pretrained model

    model_name = model_name # could be fbresnet152 or inceptionresnetv2

    if(model_name == 'senet154'):
        model = pretrainedmodels.senet154(pretrained='imagenet')
    elif(model_name == 'se_resnet152'):
        model = pretrainedmodels.se_resnet152(pretrained='imagenet')
    elif(model_name == 'se_resnext101_32x4d'):
        model = pretrainedmodels.se_resnext101_32x4d(pretrained='imagenet')
    elif(model_name == 'resnet152'):
        model = pretrainedmodels.resnet152(pretrained='imagenet')
    elif(model_name == 'resnet101'):
        model = pretrainedmodels.resnet101(pretrained='imagenet')
    elif(model_name == 'densenet201'):
        model = pretrainedmodels.densenet201(pretrained='imagenet')

    model.to(device)
    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.last_linear.in_features
    class CustomModel(nn.Module):
        def __init__(self, model):
            super(CustomModel, self).__init__()
            self.features = nn.Sequential(*list(model.children())[:-1]  )
            self.classifier = nn.Sequential(
                torch.nn.Linear(num_ftrs, 128),
                torch.nn.Dropout(0.3),  # drop 50% of the neuron
                torch.nn.Linear(128, 7)
            )
        
        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x
    model = CustomModel(model)
    freeze_layer(model.features)
    num_ftrs = list(model.classifier.children())[-1].out_features

    model.to(device)
    model.name = model_name
    return model, num_ftrs


if __name__ == "__main__":
    ################## train individual model

    model_name=['senet154', 'se_resnet152', 'se_resnext101_32x4d', 'resnet152', 'resnet101']

    for model_n in model_name:
            
        

        model, num_ftrs = build_model(model_n)
        # model.name = model_n

        model = start_train(model)
        eval_model(model)


