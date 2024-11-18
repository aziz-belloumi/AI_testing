import torch
import os
import pandas as pd
import matplotlib.pyplot as plt
from torchvision.io import read_image 
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader


# preparing tarin data 
training_data = datasets.FashionMNIST(
    root = "data" ,
    train = True ,
    download = True ,
    transform = ToTensor()
)

#preparing test data
test_data = datasets.FashionMNIST(
    root = "data" ,
    train = False ,
    download = True , 
    transform =  ToTensor() 
)

# both of the data needs to be converted at first to tensors 
# ToTensor() transform will convert the pixel values into the range [0, 1] by dividing each pixel value by 255.
# This helps neural networks learn more efficiently since normalized values make training faster and more stable.

labels_map = {
    0 : "T-Shirt",
    1 : "Trouser",
    2 : "Pullover",
    3 : "Dress",
    4 : "Coat",
    5 : "Sandal",
    6 : "Shirt",
    7 : "Sneaker",
    8 : "Bag",
    9 : "Ankle Boot"
}

figure = plt.figure(figsize=(8,8))
c , r = 3 , 3
for i in range(1 , c * r + 1) :
    sample_idx = torch.randint(len(training_data) , size = (1,)).item()
    img , label = training_data[sample_idx]
    figure.add_subplot(r , c , i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze() , cmap = "gray")
plt.show()
 
 
class CustomTmageDataset(Dataset) : 
    def __init__(self , annotations_file , img_dir , transfrom = None , target_transform = None) :
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transfrom
        self.target_tanasform = target_transform
    
    def __len__(self) :
        return len(self.img_labels)
    
    def __fetitem__(self , idx) : 
        img_path = os.path.join(self.img_dir , self.img_labels.iloc[idx , 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx , 1]
        if self.transform : 
            image = self.transform(image)
        if self.target_tanasform :
            label = self.target_transform(label)
        return image , label

# training data with dataloaders
print(f"training data is stored in: {os.path.abspath(training_data.root)}")
print(f"test data is stored in: {os.path.abspath(test_data.root)}")


train_dataloader = DataLoader(training_data , batch_size = 64 , shuffle = True) 
test_datloader =  DataLoader(test_data , batch_size = 64 , shuffle = True)




train_features , train_labels = next(iter(train_dataloader))
print(f"Feature batch shape : {train_features.size()}")
print(f"labels batch shape : {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img , cmap="gray")
plt.show()
print(f"Label : {label}")


print(img)
print(label)
