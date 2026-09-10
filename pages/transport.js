// Route the existing studio API to an isolated browser worker. No uploaded model leaves this device.
const worker=new Worker(new URL('./worker.js',import.meta.url));
const pending=new Map();let sequence=0;const originalFetch=window.fetch.bind(window);
worker.onmessage=({data})=>{
 if(data.progress){const el=document.getElementById('model-subtitle');if(el)el.textContent=data.progress;return;}
 const done=pending.get(data.id);if(!done)return;pending.delete(data.id);
 let body=data.binary?Uint8Array.from(atob(data.binary),c=>c.charCodeAt(0)):JSON.stringify(data.json);
 done(new Response(body,{status:data.status,headers:{'Content-Type':data.mime||'application/json'}}));
};
worker.onerror=e=>{for(const done of pending.values())done(new Response(JSON.stringify({detail:e.message||'浏览器运行时加载失败，请刷新重试'}),{status:500}));pending.clear();};
window.fetch=async(input,options={})=>{
 const path=typeof input==='string'?input:input.url;
 if(!path.startsWith('/api/'))return originalFetch(input,options);
 let data;
 if(options.body instanceof FormData){
  data={};for(const [key,value] of options.body){if(value instanceof File){data.filename=value.name;const bytes=new Uint8Array(await value.arrayBuffer());let str='';for(let i=0;i<bytes.length;i+=32768)str+=String.fromCharCode(...bytes.subarray(i,i+32768));data.file=btoa(str);}else data[key]=value;}
 }else data=options.body?JSON.parse(options.body):undefined;
 return new Promise(resolve=>{const id=++sequence;pending.set(id,resolve);worker.postMessage({id,path,data});});
};
document.addEventListener('click',async event=>{
 const link=event.target.closest('a[href^="/api/"]');if(!link)return;
 event.preventDefault();try{const response=await fetch(link.getAttribute('href'));if(!response.ok)throw Error((await response.json()).detail);const url=URL.createObjectURL(await response.blob());const a=document.createElement('a');a.href=url;a.download='MiShape-Portfolio.zip';a.click();setTimeout(()=>URL.revokeObjectURL(url),30000);}catch(e){alert(e.message);}
});
await import('./app.js');
