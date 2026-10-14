"""Check image splits, YOLO labels, image integrity and exact file leakage."""
import hashlib,json
from collections import Counter
from PIL import Image
from project_config import ROOT

def validate(root=ROOT):
    data=root/'dataset';records=json.loads((root/'selected_images.json').read_text())
    classes={r['ImageID']:r['class'] for r in records}
    assert len(records)==80 and len(classes)==80
    seen={}; report={}
    for split,n in [('train',32),('val',4),('test',4)]:
        images=sorted((data/'images'/split).glob('*.jpg'))
        counts=Counter(classes[p.stem] for p in images)
        assert counts=={'bottle':n,'coffee_cup':n},counts
        assert {p.stem for p in images}=={p.stem for p in (data/'labels'/split).glob('*.txt')}
        boxes=Counter()
        for p in images:
            with Image.open(p) as image: image.verify()
            digest=hashlib.sha256(p.read_bytes()).hexdigest()
            assert digest not in seen,(p,seen.get(digest))
            seen[digest]=str(p)
            rows=(data/'labels'/split/(p.stem+'.txt')).read_text().splitlines()
            assert rows,p
            ids=set()
            for row in rows:
                v=list(map(float,row.split()));assert len(v)==5 and v[0] in (0,1)
                assert all(0<=x<=1 for x in v[1:]) and v[3]>0 and v[4]>0
                assert v[1]-v[3]/2>=-1e-5 and v[1]+v[3]/2<=1+1e-5
                assert v[2]-v[4]/2>=-1e-5 and v[2]+v[4]/2<=1+1e-5
                ids.add(int(v[0]));boxes[int(v[0])]+=1
            assert len(ids)==1,'Image classification assumes one class per image'
            assert ids=={0 if classes[p.stem]=='bottle' else 1}
        report[split]={'images':len(images),'images_per_class':dict(counts),'boxes':dict(boxes)}
    return report

if __name__=='__main__': print(json.dumps(validate(),indent=2))
