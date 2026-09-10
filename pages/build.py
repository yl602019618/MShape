"""Build a self-contained GitHub Pages site; runtime pinned to Pyodide 0.29.4."""
from pathlib import Path
import json, shutil, urllib.request, hashlib, zipfile
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'_site'; OUT.mkdir(exist_ok=True)
shutil.copytree(ROOT/'mishape/web',OUT,dirs_exist_ok=True)
for name in ['transport.js','worker.js','bridge.py']:
 shutil.copy2(ROOT/'pages'/name,OUT/name)
for name in ['index.html','app.js']:
 p=OUT/name;s=p.read_text().replace('="/','="./').replace("'/images/","'./images/")
 if name=='index.html':
  s=s.replace('<script type="module" src="./app.js"></script>','<script type="module" src="./transport.js"></script>')
  s=s.replace('本地工作区','浏览器工作区').replace('连接本地几何引擎','首次打开将加载 WebAssembly 几何引擎，请稍候').replace('本地设计，自动记录参数','浏览器计算 · 请下载工程保存模型')
 else:s=s.replace('mishape-session-v1','mishape-pages-session-v1').replace('无法连接本地引擎','浏览器引擎加载失败').replace('请运行 MiShape/start.command，或检查终端错误','请刷新重试，检查网络及浏览器 WebAssembly 支持')
 p.write_text(s)
with zipfile.ZipFile(OUT/'engine.zip','w',zipfile.ZIP_DEFLATED) as z:
 for folder in ['mishape','aeroshape']:
  for p in (ROOT/folder).glob('*.py'):z.write(p,p.relative_to(ROOT))
 z.writestr('mishape/web/.keep','')
 z.write(ROOT/'mishape/assets/catalog.json','mishape/assets/catalog.json')
for p in (ROOT/'mishape/assets').glob('*/viewport.json'):
 dest=OUT/'assets'/p.parent.name;dest.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest/p.name)
runtime=OUT/'runtime';runtime.mkdir(exist_ok=True)
base='https://cdn.jsdelivr.net/pyodide/v0.29.4/full/'
def get(url,path,sha=None):
 if not path.exists() or (sha and hashlib.sha256(path.read_bytes()).hexdigest()!=sha):
  print('Download',path.name,flush=True)
  with urllib.request.urlopen(url,timeout=120) as r:path.write_bytes(r.read())
 if sha and hashlib.sha256(path.read_bytes()).hexdigest()!=sha:raise ValueError('Checksum mismatch: '+str(path))
for name in ['pyodide.js','pyodide.mjs','pyodide.asm.js','pyodide.asm.wasm','python_stdlib.zip','pyodide-lock.json']:get(base+name,runtime/name)
lock=json.loads((runtime/'pyodide-lock.json').read_text());needed=set()
def add(name):
 name=name.replace("_","-")
 if name in needed:return
 needed.add(name)
 for d in lock['packages'][name]['depends']:add(d)
for n in ['numpy','scipy','pydantic','pillow','fastapi']:add(n)
for n in sorted(needed):
 p=lock['packages'][n];get(base+p['file_name'],runtime/p['file_name'],p['sha256'])
extra=[]
for name,version in [('trimesh','4.11.2'),('networkx','3.4.2'),('python-multipart','0.0.20')]:
 data=json.load(urllib.request.urlopen(f'https://pypi.org/pypi/{name}/{version}/json'))
 p=next(p for p in data['urls'] if p['filename'].endswith('none-any.whl'))
 get(p['url'],runtime/p['filename'],p['digests']['sha256']);extra.append(p['filename'])
(OUT/'runtime-packages.json').write_text(json.dumps(extra))
(OUT/'.nojekyll').touch()
print('Built',OUT)
