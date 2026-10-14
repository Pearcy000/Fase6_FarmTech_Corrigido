import time,json
from pathlib import Path
import numpy as np,pandas as pd,torch
from sklearn.metrics import precision_recall_fscore_support,confusion_matrix
from PIL import Image
from project_config import ROOT, YOLO; D=ROOT/'dataset'
# Load pretrained YOLOv5n; weight file is fetched by the YOLO training process.
weights=YOLO/'yolov5n.pt'
assert weights.exists(),f'weights not found: {weights}'
model=torch.hub.load(str(YOLO),'custom',path=str(weights),source='local',force_reload=False,device='cpu')
model.conf=0.001; model.iou=.45; model.eval()
name_to_id={'bottle':0,'coffee_cup':1}
# Map pretrained COCO class ids: bottle=39; cup=41 (approximation for Open Images Coffee cup)
map_ids={39:0,41:1}
images=sorted((D/'images'/'test').glob('*.jpg'))
# warmup
_ = model(str(images[0]), size=416); lat=[]; pred_by=[]; gt_by=[]
for p in images:
    if torch.cuda.is_available():torch.cuda.synchronize()
    t=time.perf_counter(); out=model(str(p), size=416)
    if torch.cuda.is_available():torch.cuda.synchronize()
    lat.append(time.perf_counter()-t)
    arr=out.xyxy[0].detach().cpu().numpy()
    preds=[]
    for row in arr:
        x1,y1,x2,y2,conf,cid=row
        if int(cid) in map_ids: preds.append([map_ids[int(cid)],float(conf),float(x1),float(y1),float(x2),float(y2)])
    gt=[]
    W,H=Image.open(p).size
    for line in (D/'labels'/'test'/f'{p.stem}.txt').read_text().splitlines():
        c,xc,yc,w,h=map(float,line.split());gt.append([int(c),(xc-w/2)*W,(yc-h/2)*H,(xc+w/2)*W,(yc+h/2)*H])
    pred_by.append(preds);gt_by.append(gt)

def iou(a,b):
    x1=max(a[0],b[0]);y1=max(a[1],b[1]);x2=min(a[2],b[2]);y2=min(a[3],b[3]); inter=max(0,x2-x1)*max(0,y2-y1)
    aa=max(0,a[2]-a[0])*max(0,a[3]-a[1]);bb=max(0,b[2]-b[0])*max(0,b[3]-b[1]);return inter/(aa+bb-inter+1e-12)
# Matching at fixed confidence threshold 0.25, with one-to-one greedy IoU matching.
all_counts={}; all_true=[];all_pred=[]; ap_class={}
for c in range(2):
    gts={i:[g[1:] for g in gs if g[0]==c] for i,gs in enumerate(gt_by)}
    n_gt=sum(map(len,gts.values())); detections=[]
    for i,ps in enumerate(pred_by):
        for p in ps:
            if p[0]==c:detections.append((p[1],i,p[2:]))
    detections.sort(reverse=True,key=lambda x:x[0]); matched={i:set() for i in gts};tp=[];fp=[]
    for conf,i,box in detections:
        candidates=[(iou(box,g),j) for j,g in enumerate(gts[i]) if j not in matched[i]]
        best=max(candidates,default=(0,None))
        if best[0]>=.5:tp.append(1);fp.append(0);matched[i].add(best[1])
        else:tp.append(0);fp.append(1)
    tp=np.cumsum(tp);fp=np.cumsum(fp); rec=tp/max(n_gt,1);prec=tp/np.maximum(tp+fp,1)
    # Interpolated 101-point AP at IoU .50.
    ap=float(np.mean([max(prec[rec>=r],default=0) for r in np.linspace(0,1,101)])) if len(prec) else 0.0
    ap_class[str(c)]={'class':['bottle','coffee_cup'][c],'ground_truth':n_gt,'detections':len(detections),'AP50':ap}
    # P/R/F1 at confidence>=.25 greedy match separately
    pred_b=[[] for _ in gt_by]
    for i,ps in enumerate(pred_by): pred_b[i]=[p for p in ps if p[0]==c and p[1]>=.25]
    TP=FP=FN=0
    for i in range(len(gt_by)):
        gs=gts[i]; remaining=set(range(len(gs)))
        for p in sorted(pred_b[i],key=lambda x:-x[1]):
            cand=max(((iou(p[2:],gs[j]),j) for j in remaining),default=(0,None))
            if cand[0]>=.5:TP+=1;remaining.remove(cand[1])
            else:FP+=1
        FN+=len(remaining)
    all_counts[str(c)]={'TP':TP,'FP':FP,'FN':FN,'precision':TP/(TP+FP) if TP+FP else 0,'recall':TP/(TP+FN) if TP+FN else 0}
# macro P/R/F1 from classes. mAP50 mean of class AP.
P=np.mean([v['precision'] for v in all_counts.values()]); R=np.mean([v['recall'] for v in all_counts.values()]); F=np.mean([2*v['precision']*v['recall']/(v['precision']+v['recall']) if v['precision']+v['recall'] else 0 for v in all_counts.values()])
out={'model':'YOLOv5n pretrained COCO, no fine tuning','images':len(images),'confidence_threshold':.25,'IoU_threshold':.5,'per_class':all_counts,'AP50_per_class':ap_class,'mAP50':float(np.mean([v['AP50'] for v in ap_class.values()])),'macro_precision':float(P),'macro_recall':float(R),'macro_F1':float(F),'latency_ms_mean':float(np.mean(lat)*1000),'latency_ms_sd':float(np.std(lat)*1000),'device':'CPU'}
(ROOT/'baseline_coco_metrics.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
