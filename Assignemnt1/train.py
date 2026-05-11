import torch
import torch.nn as nn
from torch.utils.data import DataLoader ,random_split
from torchvision import datasets,transforms
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True



class DogsandCats(nn.Module):
    def __init__(self):
        super().__init__()

        self.features=nn.Sequential(
            nn.Conv2d(3,32,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32,64,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64,128,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classification=nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),  
            nn.Flatten(),  
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,2)
        )

    def forward(self,x):
        x = self.features(x)
        x= self.classification(x)

        return x
    

train_transformer = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2,contrast=0.2 , saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
])
validation_transformer = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
])

data_dir = r"E:\MlOps\Practices\Assignement1\kagglecatsanddogs_5340\PetImages"

full_dataset = datasets.ImageFolder(data_dir, transform=train_transformer)

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, valid_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset,batch_size=32,shuffle=True,num_workers=0)
val_loader = DataLoader(valid_dataset,batch_size=32,shuffle=False,num_workers=0)

device = torch.device("cuda"if torch.cuda.is_available() else "cpu")
model = DogsandCats().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)

epoch_num = 10 
for epoch in range (epoch_num):
    model.train()
    running_loss = 0.0

    for images , labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output,labels)
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()

    avg_train_loss = running_loss/len(train_loader)
    print(f"Epoch {epoch+1} loss ={avg_train_loss}") 

    
print(f"epoch : {epoch}   loss : {running_loss}")


torch.save(model.state_dict(), "dogs_and_cats.pth")
