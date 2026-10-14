from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
import torch
from project_config import ROOT, YOLO;D=ROOT/'dataset';REPO=YOLO
weights=ROOT/'runs/yolo_n_60ep/weights/best.pt'; model=torch.hub.load(str(REPO),'custom',path=str(weights),source='local',device='cpu');model.conf=.25;model.eval()
files=sorted((D/'images'/'test').glob('*.jpg'))
def load_font(size, bold=False):
 for candidate in [('DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf'), ('arialbd.ttf' if bold else 'arial.ttf')]:
  try: return ImageFont.truetype(candidate,size)
  except OSError: pass
 return ImageFont.load_default()
font=load_font(10)
thumbs=[]
for p in files:
 im=Image.open(p).convert('RGB');w,h=im.size;draw=ImageDraw.Draw(im)
 for line in (D/'labels'/'test'/f'{p.stem}.txt').read_text().splitlines():
  c,xc,yc,bw,bh=map(float,line.split());box=((xc-bw/2)*w,(yc-bh/2)*h,(xc+bw/2)*w,(yc+bh/2)*h)
  draw.rectangle(box,outline='#00D8E8',width=3)
 result=model(str(p),size=416);arr=result.xyxy[0].detach().cpu().numpy()
 for x1,y1,x2,y2,conf,c in arr:
  if conf<.25:continue
  draw.rectangle((x1,y1,x2,y2),outline='#FF5A36',width=3)
  label=('P B' if int(c)==0 else 'P C')+f' {conf:.2f}'
  draw.text((max(0,x1),max(0,y1)),label,fill='white',font=font,stroke_width=2,stroke_fill='#8c2d17')
 im.thumbnail((200,215));thumbs.append((p.stem,im.copy()))
W,H=840,600;canvas=Image.new('RGB',(W,H),'white');d=ImageDraw.Draw(canvas)
title=load_font(15,True)
small=load_font(11)
d.text((12,7),'YOLOv5n ajustado (60 épocas) — teste: caixa turquesa = referência; laranja = predição (P)',fill='#111827',font=title)
d.text((12,29),'Rótulos de predição: B = bottle; C = coffee cup; limiar de confiança 0,25',fill='#374151',font=small)
for i,(name,im) in enumerate(thumbs):
 x=8+(i%4)*208;y=50+(i//4)*274
 canvas.paste(im,(x,y));d.text((x,y+im.height+4),name,fill='#111827',font=small)
(ROOT/'figures').mkdir(exist_ok=True)
out=ROOT/'figures'/'predicoes_teste_yolo60.png';canvas.save(out);print(out)
