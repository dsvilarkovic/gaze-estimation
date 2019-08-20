import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt 
from scipy.io import loadmat
import numpy as np
import torch.utils.data
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
from scipy import misc
import matplotlib.patches as patches
from datetime import datetime
from scipy.misc import imshow
import multi_modal_nn as mmnn
from face_landmark_dataset import FaceLandmarksDataset
import test as test
import sendmail


feature_size = 136
output_size = 1
image_dim = 108
im_height = 60
im_width = 108

def extract_data(indices, suffix = '', include_pos = True):
  ftrs = torch.Tensor()
  ftrs = ftrs.type(torch.cuda.FloatTensor)  
  gz = torch.Tensor()
  gz = gz.type(torch.cuda.FloatTensor) 
    
  eye_reg = torch.Tensor()
  eye_reg = eye_reg.type(torch.cuda.IntTensor)
    
    
  img_loc = np.asarray([])
#   img_loc = img_loc.type(torch.cuda.FloatTensor) 
  
  for index in indices:
    data = loadmat(str(index) + suffix + '_lmarks_location_eye.mat')
    
    #Landmark features
    ftrs_single = torch.from_numpy(data['ftrs'])
    
    
    ftrs_single = ftrs_single.type(torch.cuda.FloatTensor)  

    
    ftrs = torch.cat((ftrs, ftrs_single))
    
    #Gaze features

    gz_single = torch.from_numpy(data['gz'])
    gz_single = gz_single.type(torch.cuda.FloatTensor)  

    gz_single = torch.t(gz_single)

    gz = torch.cat((gz, gz_single))
    
    
    #Eye regions should be n X 4 size
    
    eye_reg_single = torch.from_numpy(data['eye_reg'])
    eye_reg_single = eye_reg_single.type(torch.cuda.IntTensor)
    
    
    eye_reg = torch.cat((eye_reg, eye_reg_single))
    
    
    #Get image location
    
    img_loc_single = data['location']
        
    img_loc = np.concatenate((img_loc, img_loc_single))
    
  return(ftrs, gz, eye_reg, img_loc)



#Take out test data
test_indices = [404,407,410]
test_ftrs = torch.Tensor().cuda()
test_gz = torch.Tensor().cuda()

(test_ftrs, test_gz, test_eye_reg, test_img_loc) = extract_data(test_indices)

print(test_gz.shape)
print(test_ftrs.shape)

x_test = test_ftrs
y_test = test_gz

test_face_landmarks_dataset = FaceLandmarksDataset(ftrs = test_ftrs[12000:18000], eye_regions=test_eye_reg.cuda()[12000:18000], locations=test_img_loc[12000:18000],  gz = test_gz.cuda()[12000:18000], train_transforms=None, test_transforms=None, load_type='test')

(net,optimizer) = mmnn.get_net_instance()

net.load_state_dict(torch.load('log_results/multimodal_model_16_64.py'))


sendmail.sendmail_content('Testing started')

# (error, accuracy) = test_model(net, test_ftrs, test_gz)
accuracy = test.test_model(optimizer,net, test_face_landmarks_dataset, 256)

sendmail.sendmail_content('Accuracy is ' + str(accuracy))
