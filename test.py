import os
import pandas as pd
from torch.utils.data import DataLoader
from scripts.training import SolarFlSets, oversample_func

# define dataset here
datapath = '/workspace/data/hetero_data'
img_dir = {"EUV-304" : datapath + "/euv/compressed/304", 
           "HMI-CTnuum" : datapath + "/hmi/compressed/continuum/",
           "HMI-Mag" : datapath + "/hmi/compressed/mag"}

# projecct path
crr_path = os.getcwd()
save_path = crr_path + '/Results/CV/'
file_path = crr_path + '/dataset/'

train_list = [f'24image_multi_GOES_classification_Partition1.csv', 
                    f'24image_multi_GOES_classification_Partition2.csv', 
                    f'24image_multi_GOES_classification_Partition3.csv']
test_file = f'24image_multi_GOES_classification_Partition4.csv'

# hyper parameters
channel_tag = 'Het'
batch_size = 64

# train set
df_train = pd.DataFrame([], columns = ['Timestamp', 'GOES_cls', 'Label'])
for partition in train_list:
    d = pd.read_csv(file_path + partition)
    df_train = pd.concat([df_train, d])

# test set and calibration set
df_test = pd.read_csv(file_path + test_file)

# string to datetime
df_train['Timestamp'] = pd.to_datetime(df_train['Timestamp'], format = '%Y-%m-%d %H:%M:%S')
df_test['Timestamp'] = pd.to_datetime(df_test['Timestamp'], format = '%Y-%m-%d %H:%M:%S')

# training data loader
# over/under sampling
data_training, imbalance_ratio = oversample_func(df = df_train, img_dir = img_dir, channel = channel_tag, norm = True)

# validation data loader
data_testing = SolarFlSets(annotations_df = df_test, img_dir = img_dir, channel = channel_tag, normalization = True)
train_dataloader = DataLoader(data_training, batch_size = batch_size, shuffle = True) # num_workers = 0, pin_memory = True, 
test_dataloader = DataLoader(data_testing, batch_size = batch_size, shuffle = False) # num_workers = 0, pin_memory = True,  

image, target = data_training[0]
