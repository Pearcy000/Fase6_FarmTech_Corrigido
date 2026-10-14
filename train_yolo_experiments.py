"""Standalone YOLO experiments; point FARMTECH_DATA to a new dataset workspace."""
import subprocess,sys,yaml
from project_config import ROOT,YOLO
dataset=ROOT/'dataset';config=dataset/'dataset.yaml'
config.write_text(yaml.safe_dump({'path':str(dataset),'train':'images/train','val':'images/val','test':'images/test','nc':2,'names':['bottle','coffee_cup']},sort_keys=False))
(ROOT/'logs').mkdir(exist_ok=True)
for epochs in [30,60]:
    args=[sys.executable,str(YOLO/'train.py'),'--img','416','--batch-size','8','--epochs',str(epochs),'--data',str(config),'--weights','yolov5n.pt','--workers','2','--project',str(ROOT/'runs'),'--name',f'yolo_n_{epochs}ep','--seed','42','--patience','0','--device','cpu','--exist-ok']
    with (ROOT/'logs'/f'yolo{epochs}.log').open('w',encoding='utf-8') as log:
        subprocess.run(args,cwd=YOLO,stdout=log,stderr=subprocess.STDOUT,check=True)
