import os, sys, json
from tqdm import tqdm
from pprint import pprint
import numpy as np
import torch
import torch.nn as nn
from net_MRI import CDLNet_MRI
from data import getFitLoaders
from utils import ifft2, fft2, awgn
import torchvision.transforms as transforms 
import torch.utils.data as data
import h5py

model_args = {
    'num_filters': 32,
    'filter_size': 7,   
    'stride': 1,         
    'iters' : 10,         
    'tau0'  : 1e-2,      
    'adaptive': False,   
    'init': True
}

kspace_path = "singlecoil_val"
save_path = "Models/CDLNet_MRI" 
crop_size = 320

def main():
    
    start_epoch = 0
    epochs = 2000
    train_dataloader = getMRIdataloader()
    print("data loaded")
    
    ngpu = torch.cuda.device_count()
    device = torch.device("cuda:0" if ngpu > 0 else "cpu")
    model = CDLNet_MRI(**model_args)
    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    save_freq = 10


    fit(start_epoch, epochs, train_dataloader, device, model, opt, save_freq)

def fit(start_epoch, epochs, train_dataloader, device, model, opt, save_freq):
    
    save_dir = "Models/CDLNet_MRI"

    epoch = 0
    sigma = 0.02
    
    epoch = start_epoch
    
    print("starting")
    while epoch < start_epoch + epochs:
        model.train()
        psnr = 0
        t = tqdm(iter(train_dataloader), desc='Epoch'+str(epoch), dynamic_ncols=True)
        for itern, batch in enumerate(t):
            batch = batch.to(device)
            batch_noise, sigma_n = awgn(batch, sigma)
            
            batch_image = ifft2(batch).real

            opt.zero_grad()
            with torch.set_grad_enabled(True):
                # batch_hat, _ = model(batch_masked, mask)
                batch_hat, _ = model(batch, sigma_n)
                
                loss = torch.mean((batch_image - batch_hat)**2)
                loss.backward()
                opt.step()
                model.project()

            loss = loss.item()
            psnr = psnr - 10*np.log10(loss)

        psnr = psnr/(itern+1)

        print(f"train PSNR: {psnr:.6f} dB")

        if epoch % save_freq == 0:
            path = os.path.join(save_path, str(epoch) + '.ckpt')
            saveCkpt(path, model, epoch, opt)

        epoch = epoch + 1
            
def saveCkpt(path, model,epoch,opt,sched=None):
    """ Save Checkpoint.
    Saves model, optimizer, scheduler state dicts and epoch num to path.
    """
    getSD = lambda obj: obj.state_dict() if obj is not None else None
    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'opt_state_dict':   opt.state_dict(),
                'sched_state_dict': getSD(sched)
                }, path)
    
def read_kspace_file():

    kspace_list = []
    for p in os.listdir(kspace_path):
        f = h5py.File(os.path.join(kspace_path,p), 'r')
        kspace = f['kspace']
        for k in kspace:
            kspace_list.append(k)

    return kspace_list
  
class MRI_Dataset(data.Dataset):
    def __init__(self, kspace_list, transform):
        self.data = kspace_list
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):        
        return self.transform(self.data[idx])
    
def getMRIdataloader():
    kspace_transforms = transforms.Compose([transforms.ToTensor(),transforms.CenterCrop(160)])                              

    return data.DataLoader(MRI_Dataset(read_kspace_file(),kspace_transforms))

# def load_mask():
    # path = 'singlecoil_test_v2/file1001490_v2.h5'
    # f = h5py.File(path,'r')
    # mask = torch.as_tensor(f['mask'])
    # m = mask.repeat(160)[None,:].reshape(1,160,mask.shape[0])

    # return transforms.CenterCrop(160)(m)

if __name__ == "__main__":
    main()