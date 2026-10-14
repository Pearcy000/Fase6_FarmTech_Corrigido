from pathlib import Path
import pandas as pd, json
import matplotlib.pyplot as plt
from project_config import ROOT, YOLO; OUT=ROOT/'figures'; OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titleweight':'bold','figure.dpi':150})
fig,axs=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
for ep,color in [(30,'#2563eb'),(60,'#d97706')]:
 d=pd.read_csv(ROOT/f'runs/yolo_n_{ep}ep/results.csv');d.columns=[x.strip() for x in d.columns];x=d['epoch']+1
 for ax,key,title in [(axs[0,0],'metrics/precision','Precision'),(axs[0,1],'metrics/recall','Recall'),(axs[1,0],'metrics/mAP_0.5','mAP@0.50'),(axs[1,1],'metrics/mAP_0.5:0.95','mAP@0.50:0.95')]:
  ax.plot(x,d[key],label=f'{ep} épocas',color=color,lw=1.8)
  ax.set_title(title);ax.set_xlabel('Época');ax.set_ylim(0,1);ax.grid(alpha=.25)
for ax in axs.flat: ax.legend(frameon=False)
fig.suptitle('YOLOv5n — métricas na validação',fontsize=14,fontweight='bold')
fig.savefig(OUT/'curvas_yolo_validacao.png',bbox_inches='tight');plt.close(fig)
h=pd.read_csv(ROOT/'cnn/history.csv')
fig,axs=plt.subplots(1,2,figsize=(9,3.5),constrained_layout=True)
axs[0].plot(h.epoch,h.train_loss,label='Treino',color='#2563eb');axs[0].plot(h.epoch,h.val_loss,label='Validação',color='#dc2626');axs[0].set_title('CNN — cross-entropy');axs[0].set_xlabel('Época');axs[0].grid(alpha=.25);axs[0].legend(frameon=False)
axs[1].plot(h.epoch,h.train_accuracy,label='Treino',color='#2563eb');axs[1].plot(h.epoch,h.val_accuracy,label='Validação',color='#dc2626');axs[1].set_title('CNN — acurácia');axs[1].set_xlabel('Época');axs[1].set_ylim(0,1);axs[1].grid(alpha=.25);axs[1].legend(frameon=False)
fig.suptitle('CNN treinada do zero — histórico',fontsize=13,fontweight='bold');fig.savefig(OUT/'curvas_cnn.png',bbox_inches='tight');plt.close(fig)
print(*sorted(str(p) for p in OUT.glob('*.png')),sep='\n')
