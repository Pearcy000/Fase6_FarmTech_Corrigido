import time,json
from pathlib import Path
import numpy as np,torch
from PIL import Image
from torchvision import transforms
from torch import nn
from project_config import ROOT, YOLO;D=ROOT/'dataset';W=ROOT/'cnn'/'cnn_scratch.pt';torch.set_num_threads(3)
class CNN(nn.Module):
 def __init__(self):
  super().__init__();self.net=nn.Sequential(nn.Conv2d(3,16,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(16,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(32,64,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(64,128,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Dropout(.4),nn.Linear(128,2))
 def forward(self,x):return self.net(x)
m=CNN().eval();m.load_state_dict(torch.load(W,map_location='cpu',weights_only=True))
tf=transforms.Compose([transforms.Resize((128,128)),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
files=sorted((D/'images'/'test').glob('*.jpg'))
with torch.no_grad():
 for p in files[:2]:m(tf(Image.open(p).convert('RGB')).unsqueeze(0))
lat=[]
with torch.no_grad():
 for _ in range(5):
  for p in files:
   t=time.perf_counter();x=tf(Image.open(p).convert('RGB')).unsqueeze(0);m(x);lat.append(time.perf_counter()-t)
result={'mean_ms_image':float(np.mean(lat)*1000),'sd_ms':float(np.std(lat)*1000),'samples':len(lat),'input_size':128,'device':'CPU','includes':'file read, resize, tensor conversion, normalization, forward'}
(ROOT/'cnn'/'latency_e2e.json').write_text(json.dumps(result,indent=2));print(result)
