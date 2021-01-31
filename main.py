import torch
import train
import test
import numpy as np
from scipy.io import loadmat
import multi_modal_nn as mmnn
from face_landmark_dataset import FaceLandmarksDataset
import sys
import os


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

device = torch.device("cuda:0")


train_indices = [401,402,403,405]
# test_indices = [404, 407,410]
test_indices = [404,407,410]




ftrs = torch.Tensor()
gz = torch.Tensor()

(ftrs, gz, eye_reg, img_loc) = extract_data(train_indices)

test_ftrs = torch.Tensor().cuda()
test_gz = torch.Tensor().cuda()

(test_ftrs, test_gz, test_eye_reg, test_img_loc) = extract_data(test_indices)


#----------------------------------------------

(net,optimizer, scheduler) = mmnn.get_net_instance()
face_landmarks_dataset = FaceLandmarksDataset(ftrs = ftrs, eye_regions= eye_reg.cuda(), locations=img_loc,  gz = gz.cuda(), train_transforms=None, test_transforms=None)


print(torch.cuda.memory_allocated())

y = gz.cuda()




# x.requires_grad = False
y.requires_grad = False

# x = train_input
# y = train_output

id = int(sys.argv[1])
batch_size = int(sys.argv[2])




try:
#     os.mkdir('log_results/%d' % (id))
    os.mkdir('log_results/%d/models' % (id))
except OSError:
    print ("Already exists")

output_file = 'log_results/%d/multimodal_log_%d_%d.txt' % (id, id, batch_size)
output_model_file = 'log_results/%d/models/multimodal_model_%d_%d' % (id, id, batch_size)

    
net.load_state_dict(torch.load('log_results/22/models/multimodal_model_22_512_epoch_19.py'))

net = train.train_model(scheduler, optimizer,output_model_file, net, face_landmarks_dataset , 60, 1, False, batch_size = batch_size, output_file = output_file, train_id = id)
torch.save(net.state_dict(), output_model_file)

