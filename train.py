from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
import torch
from datetime import datetime
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt 
import torch.optim as optim
import torch.nn as nn
from multi_modal_nn import image_width, image_height
import os
import sendmail
import test as test
from face_landmark_dataset import FaceLandmarksDataset
import numpy as np
from scipy.io import loadmat
from sklearn.metrics import confusion_matrix




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


def train_model(optimizer,output_model_file, net, face_landmarks_dataset, epochs_count = 10, view_step = 10, include_graph = False, batch_size = 32768, output_file = 'multimodal_log.txt'):
    f = open(output_file, 'w') 
    
    net.train()
    errors_array = []
    f.write('Batch size is %d \n' % (batch_size))
    f.write('\n')
    
    pid = os.getpid()
    content = str(datetime.now()) + ': Proces ID %d training started \n' % (pid)
    sendmail.sendmail_content(content)


    test_indices = [404,407,410]
    test_ftrs = torch.Tensor().cuda()
    test_gz = torch.Tensor().cuda()

    (test_ftrs, test_gz, test_eye_reg, test_img_loc) = extract_data(test_indices)

#     print(test_gz.shape)
#     print(test_ftrs.shape)

    x_test = test_ftrs
    y_test = test_gz
    test_face_landmarks_dataset = FaceLandmarksDataset(ftrs = test_ftrs[12000:18000], eye_regions=test_eye_reg.cuda()[12000:18000], locations=test_img_loc[12000:18000],  gz = test_gz.cuda()[12000:18000], train_transforms=None, test_transforms=None, load_type='test')

    for epochs in range(epochs_count):
      total_correct = 0

#       dataset = torch.utils.data.TensorDataset(x,y)
#       trainloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
      trainloader = torch.utils.data.DataLoader(dataset=face_landmarks_dataset , batch_size=batch_size, shuffle=True)
      f.write('Batch size is %d \n' % (batch_size))
      f.write('Next batch')
      f.write('\n')
    
      torch.save(net.state_dict(), output_model_file + '_epoch_' + str(epochs) + '.py')
      then = datetime.now()

      for i, (train_batch) in enumerate(trainloader):
            
          #calculating time
          now  = datetime.now()  
          duration = now - then 
        
          seconds = duration.total_seconds()
          minutes = divmod(seconds, 60)[0]
            
            
          f.write(str(datetime.now().time()))
          f.write('Time for training passed %d minutes and %d seconds \t' % (minutes, seconds))
          #for measuring starting time
          then = datetime.now()
            
            
          (x_batch, y_batch) = train_batch
          optimizer.zero_grad()
          yhat = torch.Tensor()
        
          yhat = yhat.type(torch.cuda.FloatTensor)  
            
            
          yhat = net(x_batch) 

          loss = nn.BCELoss()
            
#           print('tip', type(yhat), 'tip2', type(y_batch), 'tip3 ' , type(x_batch))
#           print(yhat.shape, y_batch.shape)
          output_loss = loss(yhat, y_batch)

          output_loss.backward()
          optimizer.step()
            


          yhat = yhat > 0.5
          accuracy = accuracy_score(yhat.cpu().data.numpy(), y_batch.cpu().numpy())
           
          tn, fp, fn, tp = confusion_matrix(yhat.cpu().numpy(), y_batch.cpu()).ravel()
          total_correct = total_correct + tp + tn
        
          f.write('Total correct predicted number is %d out of %d' % (tp + tn, batch_size))
            

          f.write('This batch accuracy is %f %% \t' % (100.0 * accuracy))
            
        
          f.write('Completed %d/%d  %f %%, epoch: %d \n' % (batch_size*i, len(face_landmarks_dataset) , 
                                            100.0 * batch_size*i / len(face_landmarks_dataset), epochs))


            

      if(epochs % view_step == 0):
        pid = os.getpid()
        
        epoch_accuracy = 100.0 * total_correct / len(face_landmarks_dataset)
        content = 'Proces ID %d: Epoch %d loss is %f accuracy is: %f \n' % (pid, epochs, output_loss.item(), epoch_accuracy)

        test_accuracy = test.test_model(optimizer,net, test_face_landmarks_dataset, 256)

        sendmail.sendmail_content(content + ' \t | ' + ' Test accuracy is ' + str(test_accuracy))
        f.write(content + ' \t | ' + ' Test accuracy is ' + str(test_accuracy))

        f.write(content)
      
      errors_array.append(output_loss.item())

    if(include_graph):
      plt.xlabel('Epochs')
      plt.ylabel('Errors')
      plt.plot(errors_array, label='Error')
    
    f.write(net)
    return net
  
