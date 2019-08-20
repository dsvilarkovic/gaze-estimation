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

def train_model(optimizer,output_model_file, net, face_landmarks_dataset, epochs_count = 10, view_step = 10, include_graph = False, batch_size = 32768, output_file = 'multimodal_log.txt'):
    f = open(output_file, 'w') 
    
    net.train()
    errors_array = []
    f.write('Batch size is %d \n' % (batch_size))
    f.write('\n')
    
    pid = os.getpid()
    content = str(datetime.now()) + ': Proces ID %d training started \n' % (pid)
    sendmail.sendmail_content(content)

    for epochs in range(epochs_count):
#       dataset = torch.utils.data.TensorDataset(x,y)
#       trainloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
      trainloader = torch.utils.data.DataLoader(dataset=face_landmarks_dataset , batch_size=batch_size, shuffle=True)
      f.write('Batch size is %d \n' % (batch_size))
      f.write('Next batch')
      f.write('\n')

      torch.save(net.state_dict(), output_model_file)

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
            
            

          f.write('This batch accuracy is %f %% \t' % (100.0 * accuracy))
            
        
          f.write('Completed %d/%d  %f %%, epoch: %d \n' % (batch_size*i, len(face_landmarks_dataset) , 
                                            100.0 * batch_size*i / len(face_landmarks_dataset), epochs))
            

      if(epochs % view_step == 0):
        pid = os.getpid()
        content = 'Proces ID %d: Epoch %d loss is %f accuracy is: %f \n' % (pid, epochs, output_loss.item(), accuracy)
        sendmail.sendmail_content(content)
        f.write(content)
      
      errors_array.append(output_loss.item())

    if(include_graph):
      plt.xlabel('Epochs')
      plt.ylabel('Errors')
      plt.plot(errors_array, label='Error')
    
    f.write(net)
    return net
  
