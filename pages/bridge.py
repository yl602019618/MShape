import sys, types, json, base64, pathlib, asyncio, inspect
# Numba JIT is unavailable in WASM; execute the same numerical functions in Python.
numba=types.ModuleType('numba')
numba.njit=lambda *a,**kw: a[0] if a and callable(a[0]) else lambda f:f
sys.modules['numba']=numba
import mishape.app as app
app.asyncio=asyncio
# Batches yield after each vehicle, keeping status/cancellation available in a worker.
source=inspect.getsource(app.run_batch).replace('def run_batch(', 'async def run_batch(')
source=source.replace('job["completed"] = i + 1','job["completed"] = i + 1\n            await asyncio.sleep(0.01)')
exec(source,app.__dict__)
class Executor:
 def submit(self,fn,*args):return asyncio.create_task(fn(*args))
app.EXECUTOR=Executor()

def dispatch(path,data):
 parts=path.strip('/').split('/')
 if path=='/api/bootstrap':return app.bootstrap()
 if path=='/api/health':return app.health()
 if path=='/api/models/open':return app.open_asset(data)
 if path=='/api/generate':return app.generate_vehicle(data)
 if path=='/api/import':
  name=data.pop('filename');raw=base64.b64decode(data.pop('file'));ext=name.rsplit('.',1)[-1].lower()
  if ext=='json':
   saved=json.loads(raw);model=saved.get('base_model',saved.get('model',saved))
   return app.result_model(app.save_model(model,saved.get('name',name)),saved.get('recipe'))
  model=app.import_mesh(raw,ext,data.get('up_axis','z'),data.get('front_axis','-x'),float(data.get('length_m',4.43)),data.get('units','fit'))
  return app.result_model(app.save_model(model,name))
 if parts[1]=='models':
  mid=parts[2]
  if len(parts)==3:return app.model_info(mid)
  action=parts[3]
  if action=='deform':return app.preview(mid,app.Recipe(**data))
  if action=='calibrate':return app.calibrate_model(mid,app.Calibration(**data))
  if action=='export':return app.export_model(mid,app.Export(**data))
 if parts[1]=='batches':
  if len(parts)==2:return app.batch(app.Batch(**data))
  jid=parts[2]
  if len(parts)==3:return app.batch_status(jid)
  if parts[3]=='variants':return app.batch_variant(jid,int(parts[4]))
  if parts[3]=='download':return app.batch_download(jid)
  if parts[3]=='cancel':return app.cancel_batch(jid)
 raise ValueError('Unknown browser API: '+path)

def request(payload):
 try:
  args=json.loads(payload);result=dispatch(args['path'],args.get('data') or {})
  if isinstance(result,app.FileResponse):
   body=pathlib.Path(result.path).read_bytes();mime='application/json' if str(result.path).endswith('.json') else 'application/octet-stream'
  elif isinstance(result,app.Response):body=result.body;mime=result.media_type
  else:return json.dumps({'status':200,'json':result},ensure_ascii=False,allow_nan=False)
  return json.dumps({'status':200,'binary':base64.b64encode(body).decode(),'mime':mime})
 except Exception as e:
  import traceback;traceback.print_exc()
  return json.dumps({'status':getattr(e,'status_code',422),'json':{'detail':str(getattr(e,'detail',e))}})
