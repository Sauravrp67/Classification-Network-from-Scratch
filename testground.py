from pytorch_dataloader import Dataloader
from tqdm import tqdm

dataloader_train = tqdm(Dataloader("./Data_Pytorch",batch_size = 64,random_seed  = 42,shuffle=True,valid_size= 0.1,test=False).get_data_loader())

len(dataloader_train)

