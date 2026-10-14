import json,time,random
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score,precision_recall_fscore_support,confusion_matrix,classification_report
from project_config import ROOT, YOLO; D=ROOT/'dataset'; OUT=ROOT/'cnn'; OUT.mkdir(exist_ok=True)
SEED=42; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.set_num_threads(3)
DEVICE=torch.device('cpu')
classes=['bottle','coffee_cup']
train_tf=transforms.Compose([transforms.Resize((128,128)),transforms.RandomHorizontalFlip(),transforms.RandomRotation(8),transforms.ColorJitter(brightness=.12,contrast=.12),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
eval_tf=transforms.Compose([transforms.Resize((128,128)),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
class ImgDS(Dataset):
 def __init__(self,split,tf):
  self.items=[]; self.tf=tf
  for p in sorted((D/'images'/split).glob('*.jpg')):
   lines=(D/'labels'/split/f'{p.stem}.txt').read_text().splitlines()
   labels={int(line.split()[0]) for line in lines}
   if len(labels)!=1: raise ValueError(f'CNN requires exactly one image class: {p}')
   c=labels.pop(); self.items.append((p,c))
 def __len__(self): return len(self.items)
 def __getitem__(self,i):
  p,y=self.items[i]; return self.tf(Image.open(p).convert('RGB')),y
train=ImgDS('train',train_tf); val=ImgDS('val',eval_tf); test=ImgDS('test',eval_tf)
tr=DataLoader(train,batch_size=8,shuffle=True,num_workers=0); va=DataLoader(val,batch_size=8,shuffle=False,num_workers=0); te=DataLoader(test,batch_size=8,shuffle=False,num_workers=0)
class CNN(nn.Module):
 def __init__(self):
  super().__init__(); self.net=nn.Sequential(nn.Conv2d(3,16,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(16,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(32,64,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(64,128,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Dropout(.4),nn.Linear(128,2))
 def forward(self,x): return self.net(x)
model=CNN().to(DEVICE); criterion=nn.CrossEntropyLoss(); opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0001)
def pass_epoch(dl,training):
 model.train(training); losses=[]; ys=[]; ps=[]
 for x,y in dl:
  x,y=x.to(DEVICE),y.to(DEVICE)
  if training: opt.zero_grad()
  with torch.set_grad_enabled(training):
   z=model(x); loss=criterion(z,y)
   if training: loss.backward(); opt.step()
  losses.append(loss.item()*len(y)); ys.extend(y.numpy().tolist()); ps.extend(z.argmax(1).detach().numpy().tolist())
 return float(sum(losses)/len(dl.dataset)),accuracy_score(ys,ps),np.array(ys),np.array(ps)
hist=[];best=-1;best_loss=1e9;best_state=None;best_epoch=None;stale=0;start=time.perf_counter()
for e in range(1,51):
 tl,ta,_,_=pass_epoch(tr,True); vl,vacc,_,_=pass_epoch(va,False)
 hist.append({'epoch':e,'train_loss':tl,'train_accuracy':ta,'val_loss':vl,'val_accuracy':vacc})
 print(f'epoch {e}: train_loss {tl:.4f} train_acc {ta:.3f} val_loss {vl:.4f} val_acc {vacc:.3f}',flush=True)
 if vacc>best or (vacc==best and vl<best_loss): best=vacc;best_loss=vl;best_epoch=e;best_state={k:v.clone() for k,v in model.state_dict().items()};stale=0
 else: stale+=1
 if stale>=10: print('early stopping',flush=True);break
train_seconds=time.perf_counter()-start;model.load_state_dict(best_state);torch.save(model.state_dict(),OUT/'cnn_scratch.pt')
pd.DataFrame(hist).to_csv(OUT/'history.csv',index=False)
test_loss,acc,y,p=pass_epoch(te,False);pr,re,f1,_=precision_recall_fscore_support(y,p,average='macro',zero_division=0)
cm=confusion_matrix(y,p).tolist(); report=classification_report(y,p,target_names=classes,output_dict=True,zero_division=0)
# Model-only latency; end-to-end latency is measured by measure_cnn.py.
model.eval(); x=torch.randn(1,3,128,128); 
with torch.no_grad():
 for _ in range(15): model(x)
lat=[]
with torch.no_grad():
 for im,_ in te:
  for one in im:
   t=time.perf_counter();model(one.unsqueeze(0));lat.append(time.perf_counter()-t)
result={'best_epoch':best_epoch,'epochs_run':len(hist),'train_seconds':train_seconds,'test_loss':test_loss,'test_accuracy':acc,'precision_macro':pr,'recall_macro':re,'f1_macro':f1,'confusion_matrix':cm,'classification_report':report,'latency_ms_mean':float(np.mean(lat)*1000),'latency_ms_sd':float(np.std(lat)*1000),'device':'CPU'}
json.dump(result,open(OUT/'metrics.json','w'),indent=2);print('RESULT',json.dumps(result),flush=True)
