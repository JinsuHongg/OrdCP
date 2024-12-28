import torch
# All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.nn as nn 
import torchvision.models as models


class ResNet50(nn.Module):
    def __init__(self, freeze = False) -> None:
        super(ResNet50, self).__init__()

        # load pretrained architecture from pytorch
        torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained = True)
        self.model.fc = nn.Linear(2048, 4) # for binary classification
        
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

            #unfreeze only last fully connected layers.
            for param in self.model.fc.parameters():
                param.requires_grad = True
        
        self.prob_fn = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return self.prob_fn(x)