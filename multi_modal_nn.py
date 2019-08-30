import torch
import torch.nn as nn
import torch.optim as optim


feature_size = 136
output_size = 1
image_unit_out = 1024
# image_height = 54
image_height = 56
# image_width = 108
image_width = 112
optimizer = None

class LandmarkUnit(nn.Module):
    def __init__(self, ):
        super(LandmarkUnit, self).__init__()

        self.fc1 = nn.Linear(feature_size, int(feature_size/2))  
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(int(feature_size/2), int(feature_size/2))

        self.fc3 = nn.Linear(int(feature_size/2), int(feature_size/4))

        self.fc5 = nn.Linear(int(feature_size/4), output_size)
#         self.fc5 = nn.Linear(int(feature_size/4), int(feature_size/4))


        self.out = nn.Sigmoid()


    def forward(self, x):
#         x[0:135:2] = x[0:135:2] / 608.0
#         x[1:136:2] = x[1:136:2] / 342.0
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc5(x)
#         x = self.out(x)
        return x


class ImageUnit(nn.Module):
    def __init__(self, image_height, image_width):
        super(ImageUnit, self).__init__()
        
        
        self.image_height = image_height
        self.image_width = image_width
     
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=(3,5), stride=1, padding= (1, 2))
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=(3,5), stride=1, padding= (1, 2))
        self.relu2 = nn.ReLU()

        self.pool1 = nn.MaxPool2d(kernel_size=(2,2))

        self.conv3 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=(3,5), stride=1, padding=(1, 2))
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=(3,5), stride=1, padding=(1, 2))
        
        
        self.relu4 = nn.ReLU()

        self.conv5 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=(3,5), stride=1, padding=(1, 2))
        self.relu5 = nn.ReLU()
        
        self.conv6 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=(3,5), stride=1, padding=(1, 2))
        self.relu6 = nn.ReLU()
        
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2))

#         self.fc = nn.Linear(in_features=image_dim/2 * image_dim/2 * 12, out_features=1)
    
#         self.fc = nn.Linear(in_features=image_dim/2 * image_dim/4 * 12, out_features=image_unit_out)
        self.fc = nn.Linear(in_features=self.get_final_output_size(), out_features=image_unit_out)


        self.out = nn.Sigmoid()
        
        self.dropout02 = nn.Dropout(0.2)
        self.dropout05 = nn.Dropout(0.5)
        
    def forward(self,image):
#         print('Slika-oblik', image.shape)
        output = self.conv1(image)
#         print('Slika-oblik posle conv1', output.shape)
        output = self.relu1(output)

        output = self.conv2(output)
#         print('Slika-oblik posle conv2', output.shape)

        output = self.relu2(output)

        output = self.pool1(output)
#         print('Slika-oblik posle pool', output.shape)

        output = self.conv3(output)
#         print('Slika-oblik posle conv3', output.shape)

        output = self.relu3(output)

        output = self.conv4(output)
        output = self.relu4(output)
        
        output = self.pool2(output)
        
        output = self.conv5(output)
        output = self.relu5(output)
        
        output = self.conv6(output)
        output = self.relu6(output)
        
        output = self.pool2(output)

#         print('Slika-oblik posle conv4', output.shape)


        
#         print('simple', output.shape)
        output = output.view(-1, self.get_final_output_size())
#         print('simple', output.shape, 'Dominatno')
        output = self.dropout05(output)
        output = self.fc(output)
#         print('simple', output.shape)

        return self.relu5(output)
#         return self.out(output)

    def get_final_output_size(self):
        return self.image_height/2/2/2 * self.image_width/2/2/2 * 24

class MultiModalNetwork(nn.Module):
    def __init__(self, image_height, image_width):
        super(MultiModalNetwork, self).__init__()
     
        self.image_height = image_height
        self.image_width = image_width
        #Load landmark model from file
        self.landmarkUnit = LandmarkUnit()
       
#         self.landmarkUnit.load_state_dict(torch.load('best_136_model.py'))
        self.landmarkUnit.fc5 = nn.Linear(int(feature_size/4), int(feature_size/4))
        
        #Create new image unit needed to be trained
        self.imageUnit = ImageUnit(image_height, image_width)
        
        
        
        
        self.fc1 = nn.Linear(in_features=image_unit_out + int(feature_size/4), 
                             out_features=(image_unit_out + int(feature_size/4))/ 4)
    
        self.fc2 = nn.Linear(in_features=(image_unit_out + int(feature_size/4))/ 4,
                             out_features=(image_unit_out + int(feature_size/4))/ 16)

        self.fc3 = nn.Linear(in_features=(image_unit_out + int(feature_size/4))/ 16, out_features=1)

        self.dropout = nn.Dropout(0.5)
        
        self.out = nn.Sigmoid()
        
    
    def forward(self, x):
        landmarks = x[:, 0:136]
        #SLIKA
        image = x[:, 136:].view(-1, image_height, image_width, 3) #NOVO
        image = image.permute(0,3,1,2) #NOVO

        
#         plt.imshow(image.permute(0,3,2,1).cpu().numpy()[0,:,:,:])
        
#         print('Image ', image.shape)
        
#         print('Landmarks ', landmarks.shape)

#         print(output.shape)
        imageUnitOut = self.forward_image(image)
    
        landmarksUnitOut = self.forward_landmarks(landmarks)
        
        
        print(imageUnitOut.shape, landmarksUnitOut.shape)

  
        output = torch.cat((imageUnitOut,landmarksUnitOut), dim = 1)
        
        
        output = self.dropout(output)
        output = self.fc1(output)
        
        output = self.dropout(output)
        output = self.fc2(output)
        
        output = self.dropout(output)
        output = self.fc3(output)
        
        output= self.out(output)
        
        
        return output
  
    
    def forward_image(self,image):        
        return self.imageUnit(image)

    def forward_landmarks(self, landmarks):
        return self.landmarkUnit(landmarks)


def get_net_instance():
    net = MultiModalNetwork(image_height, image_width).cuda()
    #only unfreeze last layer of landmark unit
#     for param in net.landmarkUnit.parameters():
#         param.requires_grad = False
#     net.landmarkUnit.fc5.requires_grad = True
#     net.landmarkUnit.eval()
#     net.landmarkUnit.fc5.train()

    optimizer = torch.optim.SGD(net.parameters(), lr=0.001)
    
    return (net,optimizer)


def lr_finder(optimizer, net, face_landmarks_dataset, epochs_count = 10, 
              view_step = 10, include_graph = False, batch_size = 32768, 
              lr_begin = 0.00001, lr_step = 2):

    net.train()
    
    lr_lambda = lambda x: lr_step*x
    optimizer = torch.optim.SGD(net.parameters(), lr_begin)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    
    errors_array = []
    lr_array = []
    

        
        
    for epochs in range(epochs_count):
      total_correct = 0

      trainloader = torch.utils.data.DataLoader(dataset=face_landmarks_dataset , batch_size=batch_size, shuffle=True)
    

      for i, (train_batch) in enumerate(trainloader):
           
            
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
        
          errors_array.append(output_loss)
          lr_array.append(optimizer.state_dict()["param_groups"][0]['lr'])
        
          optimizer.step()
          scheduler.step()



          yhat = yhat > 0.5
          accuracy = accuracy_score(yhat.cpu().data.numpy(), y_batch.cpu().numpy())
          
          print("Completed batch, accuracy is %f, lr is %f, loss is %f" % (accuracy, optimizer.state_dict()["param_groups"][0]['lr'], output_loss.item()))


    loss_total = list(map(lambda x: x.cpu().item(), errors_array))
      

    if(include_graph):
      plt.xlabel('Learning rates')
      plt.ylabel('Errors')
      plt.plot(lr_array, errors_array)
    




