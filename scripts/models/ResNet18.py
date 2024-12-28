import torch
# All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.nn as nn 


class ResNet18(nn.Module):
    def __init__(self, freeze = False) -> None:
        super(ResNet18, self).__init__()

        # load pretrained architecture from pytorch
        torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained = True)
        self.model.fc = nn.Linear(512 , 4) #* torchvision.models.resnet.BasicBlock.expansion
        
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