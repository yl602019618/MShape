import {loadPyodide} from '../_site/runtime/pyodide.mjs';
import fs from 'node:fs';
const dir=new URL('../_site/',import.meta.url);const py=await loadPyodide({indexURL:new URL('runtime/',dir).pathname});
await py.loadPackage(['numpy','scipy','pydantic','pillow','fastapi']);
for(const wheel of JSON.parse(fs.readFileSync(new URL('runtime-packages.json',dir),'utf8')))py.unpackArchive(new Uint8Array(fs.readFileSync(new URL('runtime/'+wheel,dir))),'zip',{extractDir:'/home/pyodide/packages'});
py.runPython("import sys; sys.path.insert(0, '/home/pyodide/packages')");
py.unpackArchive(new Uint8Array(fs.readFileSync(new URL('engine.zip',dir))),'zip');
py.runPython(fs.readFileSync(new URL('bridge.py',dir),'utf8'));
for(const name of ['porsche-930','porsche-carrera-4s']){const path='/home/pyodide/mishape/assets/'+name;py.FS.mkdirTree(path);py.FS.writeFile(path+'/viewport.json',fs.readFileSync(new URL('assets/'+name+'/viewport.json',dir)));}
async function req(path,data){py.globals.set('_payload',JSON.stringify({path,data}));let value=JSON.parse(await py.runPythonAsync('request(_payload)'));if(value.status!==200)throw Error(JSON.stringify(value));return value.json||value;}
console.time('open');const a=await req('/api/models/open',{asset_id:'porsche-930'});console.timeEnd('open');
console.time('deform');const d=await req(`/api/models/${a.model_id}/deform`,{parameters:{vehicle_length:100}});console.timeEnd('deform');
if(d.vertices.length!==a.model.vertices.length||!d.vertices.some((v,i)=>v!==a.model.vertices[i]))throw Error('Deformation failed');
const exported=await req(`/api/models/${a.model_id}/export`,{format:'obj',parameters:{vehicle_length:100}});if(!exported.binary)throw Error('Export failed');
console.time('generate');const g=await req('/api/generate',{body_style:'fastback',parameters:{}});console.timeEnd('generate');
const batch=await req('/api/batches',{model_id:g.model_id,count:2});await py.runPythonAsync('await asyncio.sleep(20)');const done=await req('/api/batches/'+batch.id);if(done.status!=='complete')throw Error(JSON.stringify(done));
console.log(JSON.stringify({ok:true,sourceVertices:a.model.vertices.length/3,generatedVertices:g.model.vertices.length/3,batch:done.completed}));
