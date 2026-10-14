import time,json
from pathlib import Path
import numpy as np,torch
from PIL import Image
from sklearn.metrics import precision_recall_fscore_support
from project_config import ROOT, YOLO; D=ROOT/'dataset'; TEST=D/'images'/'test'
images=sorted(TEST.glob('*.jpg')); gt_by=[]
for p in images:
 W,H=Image.open(p).size; g=[]
 for line in (D/'labels'/'test'/f'{p.stem}.txt').read_text().splitlines():
  c,xc,yc,w,h=map(float,line.split());g.append([int(c),(xc-w/2)*W,(yc-h/2)*H,(xc+w/2)*W,(yc+h/2)*H])
 gt_by.append(g)
def iou(a,b):
 x1=max(a[0],b[0]);y1=max(a[1],b[1]);x2=min(a[2],b[2]);y2=min(a[3],b[3]); inter=max(0,x2-x1)*max(0,y2-y1)
 aa=max(0,a[2]-a[0])*max(0,a[3]-a[1]);bb=max(0,b[2]-b[0])*max(0,b[3]-b[1]);return inter/(aa+bb-inter+1e-12)
def evaluate(name,weights,baseline=False):
 m=torch.hub.load(str(YOLO),'custom',path=str(weights),source='local',force_reload=False,device='cpu');m.eval();m.conf=.001;m.iou=.45
 pred_by=[];lat=[]
 _=m(str(images[0]),size=416)
 for p in images:
  if torch.cuda.is_available():torch.cuda.synchronize()
  t=time.perf_counter();res=m(str(p),size=416)
  if torch.cuda.is_available():torch.cuda.synchronize()
  lat.append(time.perf_counter()-t); rows=[]
  for x1,y1,x2,y2,conf,cls in res.xyxy[0].detach().cpu().numpy():
   cls=int(cls)
   if baseline:
    if cls not in (39,41):continue
    cls=0 if cls==39 else 1
   if cls in (0,1):rows.append([cls,float(conf),float(x1),float(y1),float(x2),float(y2)])
  pred_by.append(rows)
 per={}; aps=[]
 for c in range(2):
  gt=[ [g[1:] for g in gs if g[0]==c] for gs in gt_by ]; n=sum(map(len,gt)); detections=sorted([(p[1],i,p[2:]) for i,ps in enumerate(pred_by) for p in ps if p[0]==c],reverse=True)
  matched=[set() for _ in images];tps=[];fps=[]
  for conf,i,box in detections:
   cand=max(((iou(box,g),j) for j,g in enumerate(gt[i]) if j not in matched[i]),default=(0,None))
   if cand[0]>=.5:tps.append(1);fps.append(0);matched[i].add(cand[1])
   else:tps.append(0);fps.append(1)
  tp=np.cumsum(tps);fp=np.cumsum(fps);rec=tp/max(n,1);prec=tp/np.maximum(tp+fp,1)
  ap=float(np.mean([max(prec[rec>=r],default=0) for r in np.linspace(0,1,101)])) if len(prec) else 0
  aps.append(ap)
  TP=FP=FN=0
  for i in range(len(images)):
   rem=set(range(len(gt[i])))
   for p in sorted([p for p in pred_by[i] if p[0]==c and p[1]>=.25],key=lambda p:-p[1]):
    cand=max(((iou(p[2:],gt[i][j]),j) for j in rem),default=(0,None))
    if cand[0]>=.5:TP+=1;rem.remove(cand[1])
    else:FP+=1
   FN+=len(rem)
  per[['bottle','coffee_cup'][c]]={'ground_truth':n,'TP':TP,'FP':FP,'FN':FN,'precision':TP/(TP+FP) if TP+FP else 0,'recall':TP/(TP+FN) if TP+FN else 0,'AP50':ap}
 P=float(np.mean([x['precision'] for x in per.values()]));R=float(np.mean([x['recall'] for x in per.values()]));F=float(np.mean([2*x['precision']*x['recall']/(x['precision']+x['recall']) if x['precision']+x['recall'] else 0 for x in per.values()]))
 # Benchmark the same file-to-prediction path for three warmed-up passes.
 lat=[]
 for _ in range(3):
  for p in images:
   if torch.cuda.is_available():torch.cuda.synchronize()
   t=time.perf_counter();m(str(p),size=416)
   if torch.cuda.is_available():torch.cuda.synchronize()
   lat.append(time.perf_counter()-t)
 return {'model':name,'per_class':per,'macro_precision_conf025':P,'macro_recall_conf025':R,'macro_F1_conf025':F,'mAP50_101point':float(np.mean(aps)),'latency_ms_mean':float(np.mean(lat)*1000),'latency_ms_sd':float(np.std(lat)*1000),'n_images':len(images),'conf_threshold_for_PR':.25,'IoU_threshold':.5,'img_size':416,'device':'CPU'}
configs=[('COCO pretrained zero-shot',YOLO/'yolov5n.pt',True),('fine-tuned 30 ep',ROOT/'runs/yolo_n_30ep/weights/best.pt',False),('fine-tuned 60 ep',ROOT/'runs/yolo_n_60ep/weights/best.pt',False)]
results=[evaluate(*x) for x in configs]
(ROOT/'detector_comparison.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))
