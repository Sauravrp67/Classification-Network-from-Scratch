from pytorch_dataloader import Dataloader
from Nvidia_Dali_dataloader import Nvidia_DALI_Dataloader 
from model_vgg import VGG16
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from torch.nn import CrossEntropyLoss
import time
import numpy as np
import os
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_loader,valid_loader = Dataloader(data_dir='./Data_pytorch/Train',batch_size = 64,random_seed = 42,shuffle = True,valid_size= 0.1,test = False).get_data_loader()
test_loader = Dataloader(data_dir = './Data_pytorch/Test',batch_size = 64,test =True).get_data_loader()

train_loader_dali,valid_loader_dali,test_loader_dali = Nvidia_DALI_Dataloader(root_dir='./Cifar10Images',batch_size = 64,train_ratio = 0.9,shuffle = True,seed = 42).get_dali_loader()

model_vgg = model_vgg = VGG16(num_classes = 10).to(device,non_blocking=True)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_vgg.parameters(),lr = 0.001,weight_decay = 0.005)
num_classes = 10
num_epochs = 35
batch_size = 16
learning_rate = 0.001

folders = ['Model','Images']

base_dir = '/kaggle/working'

for folder in folders:
    path = os.path.join(base_dir,folder)
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created folder: {path}")
    else:
        print(f"Folder already exists: {path}")


train_losses = []
train_accuracies = []
valid_losses = []
valid_accuracies = []
epoch_times = []
best_acc = 0


# with profile(activities = [
#     ProfilerActivity.CPU],record_shapes=True) as prof:
#     with record_function("model_training"):
for epoch in range(num_epochs):
    epoch_start_time = time.time()
    model_vgg.train()
    running_loss = 0.0
    running_accuracy = 0.0
    epoch_loss = 0
    progress_bar = tqdm(train_loader_dali, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True,position=0,dynamic_ncols = True)
    data_loading_time_start = time.time()
    for i, data in enumerate(progress_bar):
        images = data[0]['data'].to(device,non_blocking=True)
        labels = data[0]['label'].to(device,non_blocking=True).squeeze(dim=1).long()
        
        outputs = model_vgg(images)
        data_loading_time_elasped = time.time() - data_loading_time_start
        predictions = torch.argmax(outputs.to('cpu',non_blocking=True),dim=1)
        loss = criterion(outputs, labels)
        accuracy = accuracy_score(labels.to('cpu',non_blocking=True),predictions)
        backward_pass_time = time.time()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        backpass_elasped = time.time() - backward_pass_time
        running_loss += loss.item()
        running_accuracy += accuracy
        avg_loss = running_loss / (i + 1)
        avg_accuracy = running_accuracy / (i + 1)
        # update progress bar description
        progress_bar.set_postfix({'Train_Loss' :f'{loss.item():.4f}', 'Train_Accuracy' : f'{accuracy * 100}%', 'Running_Loss:' :f'{avg_loss:.4f}' , 'Running Accuracy:': f'{avg_accuracy * 100}','Data Loading and Forward Pass Time elasped:':f'{data_loading_time_elasped}','BackPass time Comsumed':f'{backpass_elasped}'})
        # progress_bar.set_postfix({'Running_Loss': f'{avg_loss:.4f}'})
    progress_bar.close()
    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = running_accuracy / len(train_loader)
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_accuracy)
    # print final loss for the epoch
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f} Accuracy: {epoch_accuracy * 100}")
    progress_bar_valid = tqdm(valid_loader_dali,desc=f"Epoch {epoch+1}/{num_epochs}",leave=True,position = 0,dynamic_ncols=True)
    # Validation phase
    model_vgg.eval()
    running_loss_valid = 0
    running_accuracy_valid = 0
    
    with torch.no_grad():
        for i, data in enumerate(progress_bar_valid):
            images = data[0]['data'].to(device,non_blocking=True)
            labels = data[0]['label'].to(device,non_blocking=True).squeeze(dim=1).long()
            outputs = model_vgg(images)
            predictions = torch.argmax(outputs.cpu(), dim=1)
            loss = criterion(outputs,labels)
            accuracy = accuracy_score(labels.cpu(),predictions)
            running_loss_valid += loss.item()
            running_accuracy_valid += accuracy
            avg_loss = running_loss_valid / (i + 1)
            avg_accuracy = running_accuracy_valid / (i + 1)
            progress_bar_valid.set_postfix({'Valid_Loss' :f'{loss.item():.4f}', 'valid_Accuracy' : f'{accuracy * 100}%','Running Accuracy' : f'{avg_accuracy * 100}%','Running_Loss' : f'{avg_loss}'})
        avg_loss = running_loss_valid / len(valid_loader_dali)
        avg_accuracy = running_accuracy_valid / len(valid_loader_dali)
        valid_losses.append(avg_loss)
        valid_accuracies.append(avg_accuracy)
        print(f'Validation Accuracy on 5000 images: {100 * avg_accuracy:.2f} % Loss: {avg_loss}')
    progress_bar_valid.close()
    if epoch_accuracy > best_acc:
        print(f"New Best Accuracy : {epoch_accuracy * 100}%\nSaving best accuracy Model......")
        best_acc = epoch_accuracy
        torch.save({
            'epoch':epoch,
            'model_state_dict':model_vgg.module.state_dict(),
            'optimizer_state_dict':optimizer.state_dict(),
            'loss':epoch_loss,
            'accuracy':epoch_accuracy
        },"/kaggle/working/Model/best_model_vgg.pth")
    epoch_end_time = (time.time() - epoch_start_time) / 60
    epoch_times.append(epoch_end_time)
    print(f"Epoch Time Lasped: {epoch_end_time}")
    

            
                
### Plot the results:
plt.figure(figsize=(10,5))
plt.plot(train_losses,label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10,5))
plt.plot(train_accuracies,label='Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy over Epochs')
plt.legend()
plt.grid(True)
plt.show()
