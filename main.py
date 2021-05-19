import os, sys, json
import cv2
from tqdm import tqdm
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from net import CDLNet, CDLNet_I
from data import getFitLoaders
from utils import awgn, imgLoad, mask, block_mask

# Arguments
model_args = {
    'num_filters': 32,
    'filter_size': 7,   
    'stride': 2,         
    'iters' : 10,         
    'tau0'  : 1e-2,      
    'adaptive': False,   
    'init': True
}

loader_args = {
    'batch_size': 10,
    'crop_size': 128,
    'trn_path_list': ['CBSD432'],
    'val_path_list': ['Kodak'],
    'tst_path_list': ['CBSD68']
}

fit_args = {
    'epochs': 6000,
    'noise_std': 25,
    'val_freq': 50,
    'save_freq': 5,
    'backtrack_thresh': 2,
    'verbose': True,
    'clip_grad': 0.05
}


save_dir = 'Models/CDLNet-test-I2'

# Device
ngpu = torch.cuda.device_count()
device = torch.device("cuda:0" if ngpu > 0 else "cpu")
print(device)

def test_batch(test_loader, model_args, level,type):
  if type=='denoising':
    ckpt = torch.load(load_path, map_location=torch.device('cpu'))
    test_model = CDLNet(**model_args)
    test_model.to(device)
    test_model.load_state_dict(ckpt["model_state_dict"])
    test_model.eval()
    test_psnr, test_loss = val_epoch_d(1, test_model, test_loader, level)
    return test_psnr, test_loss
  elif type=='inpainting':
    ckpt = torch.load(load_path, map_location=torch.device('cpu'))
    test_model = CDLNet_I(**model_args)
    test_model.to(device)
    test_model.load_state_dict(ckpt["model_state_dict"])
    test_model.eval()
    test_psnr, test_loss = val_epoch_I(1, test_model, test_loader, level)
    return test_psnr, test_loss

def test_single_image(img_path, load_path, model_args, level, type):
  ckpt = torch.load(load_path, map_location=torch.device('cpu'))
  x = imgLoad(img_path, gray=True).to(device)
  z = x
  x_hat = x

  if type=='inpainting':
    test_model = CDLNet_I(**model_args)
    test_model.to(device)
    test_model.load_state_dict(ckpt["model_state_dict"])
    test_model.eval()

    z, m = mask(x, level)
    D = test_model.D.weight.cpu()

    with torch.no_grad():
        x_hat,sc = test_model(z,m)

  elif type=='denoising':
    test_model = CDLNet(**model_args)
    test_model.to(device)
    test_model.load_state_dict(ckpt["model_state_dict"])
    test_model.eval()

    D = test_model.D.weight.cpu()
    z, sigma_n = awgn(x, level)

    with torch.no_grad():
        x_hat,sc = test_model(z,sigma_n)  

  plt.figure(figsize=(20,20))
  plt.subplot(1,3,1)
  plt.imshow(x.cpu().numpy()[0][0],'gray',vmin=0,vmax=1)
  plt.axis('off')
  plt.subplot(1,3,2)
  plt.imshow(z.cpu().numpy()[0][0],'gray',vmin=0,vmax=1)
  plt.axis('off')
  plt.subplot(1,3,3)
  plt.imshow(x_hat.cpu().numpy()[0][0],'gray',vmin=0,vmax=1)
  plt.axis('off')
  plt.tight_layout()
  return x.cpu().numpy()[0][0], z.cpu().numpy()[0][0], x_hat.cpu().numpy()[0][0]

loaders = getFitLoaders(**loader_args)
test_loader = loaders['test']

def train_epoch_d(epoch, model, loader, opt, sigma, clip_grad):
    model.train()
    psnr = 0
    t = tqdm(iter(loader), desc='train epoch:'+str(epoch), dynamic_ncols=True)
    for itern, batch in enumerate(t):
        batch = batch.to(device)
        noisy_batch, sigma_n = awgn(batch, sigma)

        opt.zero_grad()
        with torch.set_grad_enabled(True):
            batch_hat, _ = model(noisy_batch, sigma_n)
            loss = torch.mean((batch - batch_hat)**2)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            opt.step()
            model.project()
        loss = loss.item()
        psnr = psnr - 10*np.log10(loss) 
    psnr = psnr/(itern+1)
    print(f"train PSNR: {psnr:.3f} dB")

    return psnr, loss
    
def val_epoch_d(epoch, model, loader, sigma):
    model.eval()
    psnr = 0
    t = tqdm(iter(loader), desc='val epoch:'+str(epoch), dynamic_ncols=True)
    for itern, batch in enumerate(t):
        batch = batch.to(device)
        noisy_batch, sigma_n = awgn(batch, sigma)

        with torch.no_grad():
            batch_hat, _ = model(noisy_batch, sigma_n)
            loss = torch.mean((batch - batch_hat)**2)
        loss = loss.item()
        psnr = psnr - 10*np.log10(loss)
    psnr = psnr/(itern+1)
    print(f"val PSNR: {psnr:.3f} dB")

    return psnr, loss

def train_epoch_I(epoch, model, loader, opt, ber, clip_grad):
    model.train()
    psnr = 0
    t = tqdm(iter(loader), desc='train epoch:'+str(epoch), dynamic_ncols=True)
    for itern, batch in enumerate(t):
        batch = batch.to(device)
        masked_batch, m = mask(batch, ber)

        opt.zero_grad()
        with torch.set_grad_enabled(True):
            batch_hat, _ = model(masked_batch, m)
            loss = torch.mean((batch - batch_hat)**2)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            opt.step()
            model.project()
        loss = loss.item()
        psnr = psnr - 10*np.log10(loss) 
    psnr = psnr/(itern+1)
    print(f"train PSNR: {psnr:.3f} dB")

    return psnr, loss
    
def val_epoch_I(epoch, model, loader, ber):
    model.eval()
    psnr = 0
    t = tqdm(iter(loader), desc='val epoch:'+str(epoch), dynamic_ncols=True)
    for itern, batch in enumerate(t):
        batch = batch.to(device)
        masked_batch, m = mask(batch, ber)

        with torch.no_grad():
            batch_hat, _ = model(masked_batch, m)
            loss = torch.mean((batch - batch_hat)**2)
        loss = loss.item()
        psnr = psnr - 10*np.log10(loss)
    psnr = psnr/(itern+1)
    print(f"val PSNR: {psnr:.3f} dB")

    return psnr, loss
    
def saveCkpt(path, model=None,epoch=None,opt=None,sched=None):
    """ Save Checkpoint.
    Saves model, optimizer, scheduler state dicts and epoch num to path.
    """
    getSD = lambda obj: obj.state_dict() if obj is not None else None
    torch.save({'epoch': epoch,
                'model_state_dict': getSD(model),
                'opt_state_dict':   getSD(opt),
                'sched_state_dict': getSD(sched)
                }, path)


def fit_model(train_epoch, val_epoch, save_path, model, loader, num_epochs, opt, level, clip_grad, sched):
  train_psnr_list = []
  train_loss_list = []
  test_psnr_list = []
  test_loss_list = []

  train_loader = loaders['train']
  val_loader = loaders['val']
  for epoch in range(num_epochs):

    train_psnr, train_loss = train_epoch(epoch, model, train_loader, opt, level, clip_grad)
    test_psnr, test_loss = val_epoch(epoch, model, val_loader, level)    
    sched.step()

    train_psnr_list.append(train_psnr)
    train_loss_list.append(train_loss)
    test_psnr_list.append(test_psnr)
    test_loss_list.append(test_loss)

    path = os.path.join(save_path, str(epoch) + '.ckpt')
    if(epoch%5 == 0):
      saveCkpt(path, model, epoch, opt, sched)

  return   train_psnr_list, train_loss_list, test_psnr_list, test_loss_list

#Fit
model_args = {
    'num_filters': 32,
    'filter_size': 7,   
    'stride': 2,         
    'iters' : 6,         
    'tau0'  : 1e-2,      
    'adaptive': False,   
    'init': True
}

save_path = 'Models/CDLNet-test2'
model = CDLNet(**model_args)
model.to(device)

opt = torch.optim.Adam(model.parameters(), lr=LR)
sched = torch.optim.lr_scheduler.StepLR(opt, gamma=0.95,step_size=50)
fit_args = {
    'num_epochs': 2000, 
    'opt': opt, 
    'level': 25 , 
    'clip_grad': 0.05,
    'sched': sched
}

loaders = getFitLoaders(**loader_args)
trpl, trll, tspl, tsll = fit_model(train_epoch_d, val_epoch_d, save_path, model, loaders, **fit_args)

for ber in ber_list:
  train_psnr_list = []
  train_loss_list = []
  test_psnr_list = []
  test_loss_list = []

  save_path = os.path.join(save_dir, str(ber))
  if not os.path.exists(save_path):
    os.makedirs(save_path)

  for epoch in range(num_epochs):

    train_psnr, train_loss = train_epoch(model, train_loader, opt, ber, clip_grad)
    test_psnr, test_loss = val_epoch(model, val_loader, ber)    
    sched.step()

    train_psnr_list.append(train_psnr)
    train_loss_list.append(train_loss)
    test_psnr_list.append(test_psnr)
    test_loss_list.append(test_loss)

    path = os.path.join(save_path, str(epoch) + '.ckpt')
    if(epoch%5 == 0):
      saveCkpt(path, model, epoch, opt, sched)

  psnr_path =  os.path.join(save_dir, 'PSNR', str(ber)+'psnr.txt')
  with open(psnr_path,'w') as psnr_file:
    for i in range(len(train_psnr_list)):
      psnr_file.write(f'{train_psnr_list[i]:.4f}  ')
      psnr_file.write(',')
      psnr_file.write(f'{test_psnr_list[i]:.4f}  ')
      psnr_file.write('\n')

  loss_path =  os.path.join(save_dir, 'LOSS', str(ber)+'loss.txt')
  with open(loss_path,'w') as loss_file:
    for i in range(len(train_loss_list)):
      loss_file.write(f'{train_loss_list[i]:.4f}  ')
      loss_file.write(',')
      loss_file.write(f'{test_loss_list[i]:.4f}  ')
      loss_file.write('\n')

train_list = []
val_list = []
for ber in ber_list:
  path = os.path.join(save_dir,'PSNR',str(ber)+'psnr.txt')
  f = open(path, 'r+')
  lines = [line for line in f.readlines()]
  f.close
  train = float(lines[49][:7])
  val = float(lines[49][10:17])
  train_list.append(train)
  val_list.append(val)

plt.plot(ber_list,train_list,'blue',label='train')
plt.plot(ber_list,val_list,'red',label = 'validation')
plt.ylim([10,50])
plt.legend(bbox_to_anchor=(0.3, 0.22))

f = open('Models/PSNR/psnr.txt', 'r+')
train = [float(line[:7]) for line in f.readlines()]
f.close

f = open('Models/PSNR/psnr.txt', 'r+')
val = [float(line[10:17]) for line in f.readlines()]
f.close

test = []
test_loader = loaders['test']
for ber_model in ber_list:
  test_one = []
  load_path = os.path.join(save_dir,str(ber_model),'50.ckpt')
  args_path = 'Models/CDLNet-S25/args.json'
  args = json.load(open(args_path))
  model_args = args['model']

  ckpt = torch.load(load_path, map_location=torch.device('cpu'))
  test_model = CDLNet_I(**model_args)
  test_model.to(device)
  test_model.load_state_dict(ckpt["model_state_dict"])
  for ber_img in ber_list:
    test_psnr, test_loss = val_epoch(test_model, test_loader, ber_img) 
    test_one.append(test_psnr)
  test.append(test_one)

