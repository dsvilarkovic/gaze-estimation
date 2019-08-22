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
        x[0:135:2] = x[0:135:2] / 608.0
        x[1:136:2] = x[1:136:2] / 342.0
        
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
     
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3,5), stride=1, padding= (1, 2))
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,5), stride=1, padding= (1, 2))
        self.relu2 = nn.ReLU()

        self.pool1 = nn.MaxPool2d(kernel_size=(2,2))

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,5), stride=1, padding=(1, 2))
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,5), stride=1, padding=(1, 2))
        
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2))

        self.relu4 = nn.ReLU()

        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,5), stride=1, padding=(1, 2))
        self.relu5 = nn.ReLU()
        
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,5), stride=1, padding=(1, 2))
        self.relu6 = nn.ReLU()

        self.pool3 = nn.MaxPool2d(kernel_size=(2,2))

        self.conv7 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,5), stride=1, padding=(1, 2))
        self.relu7 = nn.ReLU()
        
        self.conv8 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,5), stride=1, padding=(1, 2))
        self.relu8 = nn.ReLU()
        
        self.pool4 = nn.MaxPool2d(kernel_size=(2,2))

#         self.fc = nn.Linear(in_features=image_dim/2 * image_dim/2 * 12, out_features=1)
    
#         self.fc = nn.Linear(in_features=image_dim/2 * image_dim/4 * 12, out_features=image_unit_out)
        self.fc = nn.Linear(in_features=self.get_final_output_size(), out_features=image_unit_out)


        self.out = nn.Sigmoid()
        
        self.dropout02 = nn.Dropout(0.2)
        self.dropout05 = nn.Dropout(0.5)
        
    def forward(self,image):
#         print('Slika-oblik', image.shape)
        output = self.conv1(image)
        output = self.relu1(output)

        output = self.conv2(output)

        output = self.relu2(output)

        output = self.pool1(output)

        output = self.conv3(output)

        output = self.relu3(output)

        output = self.conv4(output)

        output = self.relu4(output)
        
        output = self.pool2(output)

        output = self.conv5(output)

        output = self.relu5(output)
        
        output = self.conv6(output)

        output = self.relu6(output)
        
        output = self.pool3(output)

        output = self.conv7(output)
        output = self.relu7(output)

        output = self.conv8(output)
        output = self.relu8(output)
      

        output = self.pool4(output)
        

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
        return self.image_height/2/2/2/2 * self.image_width/2/2/2/2 * 256

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







