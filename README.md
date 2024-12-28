# OrdCP
Conformal prediction with ordinal regression

# Parser arguments
"--epochs", type = int, default = 12, help = "number of epochs"  
"--batch_size", type = int, default = 64, help = "batch size"  
"--lr", type = float, default = 1e-6, help = "learning rate"  
"--weight_decay", type = list, default = [0, 1e-4], help = "regularization parameter"  
"--max_lr", type = float, default = 1e-3, help = "MAX learning rate"  
"--models", type = str, default = 'Mobilenet', help = "Enter Mobilenet, Resnet18, Resnet34, Resnet50"  
"--freeze", type = bool, default = False, help = "Enter True or False to freeze the convolutional layers"  
"--data", type = str, default = 'EUV304', help = "Enter Data source: EUV304, HMI-CTnuum, HMI-Mag, Het"  

# Running the code
python -m Main_CV

# Introduction
This study predicts Multi-class solar flare events (with ordinality) using Solar full-disk images (with multi-channels). We propose nobel conformal predictor for ordinal regression (or ordinal logistic regression). The full-disk images labeled with 0 (Flare quite and class A), 1(class B), 2(class C), and 3(class M and X). The integer labels are transformed into cumulative binary labels, as described below, which is different from one-hot encoding:  
0 --> [1, 0, 0, 0], 1 --> [1, 1, 0, 0], 2 --> [1, 1, 1, 0], 3 --> [1, 1, 1, 1]

The outputs from the models with four neurons are feeded into the sigmoid function. Cross entropy for the loss function is used.
loss =  $- \sum_i^K (t_i \times log(o_i) + (1-t_i) \times log(1-o_i))$ 