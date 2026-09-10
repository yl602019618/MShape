importScripts('./runtime/pyodide.js');
let py;
const dbReady=new Promise(resolve=>{
 const req=indexedDB.open('mishape-models-v1',1);
 req.onupgradeneeded=()=>req.result.createObjectStore('models');
 req.onsuccess=()=>resolve(req.result);req.onerror=()=>resolve(null);
});
async function cacheModel(id,model){
 const db=await dbReady;if(!db)return null;
 return new Promise(resolve=>{
  const tx=db.transaction('models',model?'readwrite':'readonly');
  const req=model?tx.objectStore('models').put(model,id):tx.objectStore('models').get(id);
  req.onsuccess=()=>resolve(req.result);req.onerror=()=>resolve(null);
 });
}
const ready=(async()=>{
 postMessage({progress:'正在启动浏览器 Python / WebAssembly…'});
 py=await loadPyodide({indexURL:new URL('./runtime/',self.location).href});
 postMessage({progress:'正在加载 NumPy / SciPy 几何计算库…'});
 await py.loadPackage(['numpy','scipy','pydantic','pillow','fastapi']);
 const wheels=await (await fetch('./runtime-packages.json')).json();
 for(const wheel of wheels)py.unpackArchive(await (await fetch('./runtime/'+wheel)).arrayBuffer(),'zip',{extractDir:'/home/pyodide/packages'});
 py.runPython("import sys; sys.path.insert(0, '/home/pyodide/packages')");
 py.unpackArchive(await (await fetch('./engine.zip')).arrayBuffer(),'zip');
 py.runPython(await (await fetch('./bridge.py')).text());
 postMessage({progress:'几何引擎就绪，正在读取车辆…'});
})();
self.onmessage=async({data:{id,path,data}})=>{
 try{
  await ready;
  if(path==='/api/models/open'){
   const dest=`/home/pyodide/mishape/assets/${data.asset_id}`;
   if(!py.FS.analyzePath(dest+'/viewport.json').exists){
    const response=await fetch(`./assets/${encodeURIComponent(data.asset_id)}/viewport.json`);
    if(!response.ok)throw Error('车辆下载失败 '+response.status);
    py.FS.mkdirTree(dest);py.FS.writeFile(dest+'/viewport.json',new Uint8Array(await response.arrayBuffer()));
   }
  }
  const match=path.match(/^\/api\/models\/([a-f0-9]{32})(?:\/|$)/);
  if(match){const dest=String(py.runPython('str(app.STATE)'))+'/'+match[1]+'.json';
   if(!py.FS.analyzePath(dest).exists){const saved=await cacheModel(match[1]);if(saved)py.FS.writeFile(dest,JSON.stringify(saved));}
  }
  py.globals.set('_payload',JSON.stringify({path,data}));
  const result=JSON.parse(await py.runPythonAsync('request(_payload)'));
  if(result.json?.model_id&&result.json?.model)await cacheModel(result.json.model_id,result.json.model);
  postMessage({id,...result});
 }catch(e){postMessage({id,status:500,json:{detail:'浏览器几何引擎：'+e.message}});}
};
