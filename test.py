from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
import torch
from datetime import datetime
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt 
import torch.optim as optim
from multi_modal_nn import optimizer
from sklearn.metrics import confusion_matrix

def test_model(optimizer,net, test_face_landmarks_dataset, batch_size = 2048):
  net.eval()
#   y_total = torch.Tensor().cuda()
#   y_hat_total = torch.Tensor().cuda()

  accuracies = []

  testloader = torch.utils.data.DataLoader(dataset=test_face_landmarks_dataset , batch_size=batch_size, shuffle=True)
  
  then = datetime.now()
  total_correct = 0

  print('Test size is %d' % (len(test_face_landmarks_dataset)))

  for i, (train_batch) in enumerate(testloader):

      #calculating time
      now  = datetime.now()  
      duration = now - then 

      seconds = duration.total_seconds()
      minutes = divmod(seconds, 60)[0]


      print(datetime.now().time())
      print('Time for testing passed %d minutes and %d seconds' % (minutes, seconds))
      #for measuring starting time
      then = datetime.now()


      (x_batch, y_batch) = train_batch
      optimizer.zero_grad()
      yhat = torch.Tensor()
      yhat = yhat.type(torch.cuda.FloatTensor)  


      yhat = net(x_batch) 


      yhat = yhat > 0.5
      accuracy = accuracy_score(yhat.cpu().numpy(), y_batch.cpu().numpy())

  


      print('This batch accuracy is %f %%' % (100.0 * accuracy) )
      print('Completed %d/%d  %f %%' % (batch_size*(i + 1), len(test_face_landmarks_dataset) , 100.0 * batch_size*(i + 1) / len(test_face_landmarks_dataset)))
        
        
      tn, fp, fn, tp = confusion_matrix(yhat.cpu().numpy(), y_batch.cpu()).ravel()
      total_correct = total_correct + tp + tn
        
      print('Total correct predicted number is %d out of %d' % (tp + tn, batch_size))
  

  return 100.0 * total_correct / len(test_face_landmarks_dataset)

  