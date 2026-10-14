"""Optional: restore exactly the selected images from their saved source metadata."""
import json, requests
from PIL import Image
from io import BytesIO
from project_config import ROOT
records=json.loads((ROOT/'selected_images.json').read_text(encoding='utf-8'))
(ROOT/'raw').mkdir(parents=True,exist_ok=True)
failed=[]
for record in records:
    dest=ROOT/'raw'/f"{record['ImageID']}.jpg"
    if dest.exists(): continue
    for key in ['Thumbnail300KURL','OriginalURL']:
        try:
            response=requests.get(record[key],timeout=30);response.raise_for_status()
            with Image.open(BytesIO(response.content)) as image:
                image.convert('RGB').save(dest,quality=95)
            break
        except Exception as error:
            print(record['ImageID'],key,type(error).__name__)
    if not dest.exists(): failed.append(record['ImageID'])
(ROOT/'download_failures.json').write_text(json.dumps(failed,indent=2))
if failed: raise RuntimeError(f'Failed downloads: {failed}')
print('Selected images restored in raw/. The supplied split dataset remains unchanged.')
