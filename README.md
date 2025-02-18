# OrdCP
Conformal prediction with ordinal regression

ordinal regression, conformal prediction, solar physics.

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
python -m Main_CV --model Mobilenet --data EUV304

# Introduction
This study predicts Multi-class solar flare events (with ordinality) using Solar full-disk images (with multi-channels). We propose nobel conformal predictor for ordinal regression (or ordinal logistic regression). The full-disk images labeled with 0 (Flare quite and class A), 1(class B), 2(class C), and 3(class M and X). The integer labels are transformed into cumulative binary labels, as described below, which is different from one-hot encoding: 0 &rarr; [1, 0, 0, 0], 1 &rarr; [1, 1, 0, 0], 2 &rarr; [1, 1, 1, 0], 3 &rarr; [1, 1, 1, 1]  

The outputs from the models with four neurons are feeded into the sigmoid function. Cross entropy for the loss function is used.
loss =  $- \sum_i^K (t_i \times log(o_i) + (1-t_i) \times log(1-o_i))$, where $o_i$ is output (probability) from the sigmoid and $t_i$ is the target label (0 or 1). 

## Mondrian CP
This implementation focuses on Mondrian Conformal Prediction, a method that provides reliable prediction intervals while maintaining class-specific coverage guarantees. The approach ensures that predictive uncertainty is accurately quantified for each individual class. The non-conformity scores are calculated using cumulative probability summation, which provides a robust measure of prediction reliability. The method proceeds as follows:

1. For each class, we compute non-conformity scores based on the cumulative probability distribution
2. These scores maintain separate calibration for each class label
3. The result is a prediction set that guarantees the desired coverage level within each class

The non-conformity score α for an example (x, y) is computed as:\
$C_{i} = \sum_{k=1}^K |y_{i,k} - \hat{h}_{i,k} (x_i)|$\
$$ C_{i} = \sum_{k=1}^{K} \lvert y_{i,k} - \hat{h}_{i,k}(x_i) \rvert, i \in X_{calibration} $$

#### Let:
1. $K$ represent the total number of classes
2. $y_{i,k}$ denote the true label for the $i$-th instance in class $k$
3. $\hat{h}_{i,k}$ represent our predicted probability for the $i$-th instance in class $k$