import torch
# All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.nn as nn
# import torchvision.models as models

class mobilenet(nn.Module):
    def __init__(self, freeze = False) -> None:
        super(mobilenet, self).__init__()

        # Load mobilenet v3
        torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v3_large', pretrained=True)
        self.model.classifier[-1] = nn.Linear(1280, 4)

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

            #unfreeze only last fully connected layers.
            for param in self.model.classifier.parameters():
                param.requires_grad = True

        self.prob_fn = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return self.prob_fn(x)