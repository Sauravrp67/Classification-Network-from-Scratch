from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch
import numpy as np
class Dataloader():
    def __init__(self,data_dir,batch_size,random_seed=42,shuffle=True,valid_size = 0.1,test=False):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.shuffle = shuffle
        self.valid_size = valid_size
        self.test = test
    
    def get_data_loader(self):
        # We need to normalize each channel with a mean and std values so that exploding gradients problem don't occur.
        # Also, the initial weights assumes (He,Xavier) which assume input features are zero-centered.
        normalize = transforms.Normalize(mean = [0.5071,0.4867,0.4408],std = [0.2675,0.2565,0.2761])
        # Resizing the image size making it suitable for VGG network.
        transform_object = [transforms.Resize((224,224)),transforms.ToTensor(),normalize]
        transform = transforms.Compose(transform_object)
        #Testing Set
        if self.test:
            print("downloading started...")
            dataset = datasets.CIFFAR10(
                root = self.data_dir,
                train = False,
                download=True,
                transform = transform
            )
            print("Downloading Completed...")
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size = self.batch_size,
                shuffle = self.shuffle,
                #Pinning Memory means storing the tensors in non-pageable memory of virtual memory,
                #Meaning they remain fixed in the RAM once loaded and OS cannot swap these segments to Seconday storage during runtime
                #This reduces the latency of transferring cpu tensor to GPU during training loop
                pin_memory = True, 
                num_workers = 2,

            )
            return dataloader
        
        train_dataset = datasets.CIFAR10(
            root=self.data_dir,
            train = True,
            download = True,
            transform = transform
        )
        num_train = len(train_dataset)
        indices = list(range(num_train))
        split = int(np.floor(self.valid_size * num_train))

        if self.shuffle:
            np.random.seed(self.random_seed)
            np.random.shuffle(indices)
        train_idx, valid_idx = indices[split:],indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,batch_size = self.batch_size,sampler = train_sampler,pin_memory = True,num_workers = 2
        )
        valid_laoder = torch.utils.data.DataLoader(
            train_dataset,batch_size = self.batch_size,sampler = valid_sampler, pin_memory = True,num_workers = 2
        )
        return (train_loader,valid_laoder)