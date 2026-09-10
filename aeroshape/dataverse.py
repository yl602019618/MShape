"""Dataverse native metadata/access adapter. Network use is CLI-only.

No whole-dataset download, no login bypass, no guessed file IDs. Download only
explicitly selected inventory IDs; stream to a temporary file and verify the
repository checksum and exact byte length before publication.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen, HTTPRedirectHandler, build_opener
from .drivaer import DOI, classify_name, safe_relative

SERVER='https://dataverse.harvard.edu'


def parse_inventory(payload:dict,*,doi=DOI) -> dict:
    if payload.get('status')!='OK':raise ValueError('Dataverse did not return OK.')
    data=payload.get('data',{});version=data.get('latestVersion',data)
    files=version.get('files')
    if not isinstance(files,list):raise ValueError('No file inventory in Dataverse metadata response.')
    published=version.get('versionState')
    rows=[]
    for entry in files:
        f=entry.get('dataFile',{});fid=f.get('id')
        if not isinstance(fid,int) or fid<=0:raise ValueError('Invalid Dataverse file ID.')
        filename=f.get('filename') or entry.get('label') or ''
        directory=entry.get('directoryLabel','')
        path=safe_relative((directory+'/' if directory else '')+filename)
        suffix=Path(filename).suffix.lower();kind='other'
        if suffix in ('.stl','.obj','.ply','.off','.glb','.gltf','.npz'):kind='surface_candidate'
        elif suffix in ('.zip','.tar','.gz','.tgz'):kind='archive_unknown_contents'
        elif suffix in ('.vtk','.vtp','.vtu'):kind='vtk_requires_explicit_surface_conversion'
        checksum=f.get('checksum') or {}
        rows.append(dict(file_id=fid,path=path,filename=filename,bytes=int(f.get('filesize',0)),
            restricted=bool(entry.get('restricted',False)),checksum=checksum,
            content_type=f.get('contentType'),kind=kind,body_type_hint=classify_name(filename)['body_type']))
    return dict(schema='aeroshape.dataverse-inventory.v1',dataset_doi=doi,
        version=f"{version.get('versionNumber','?')}.{version.get('versionMinorNumber','?')}",
        version_state=published,files=rows,total=len(rows),
        source='Dataverse metadata API',downloaded=False,
        warning='An archive inventory row is not an individual car. Review contents after download.')


def fetch_inventory(*,doi=DOI,version=':latest-published',open_url=None) -> dict:
    if not version.replace('.','').isdigit() and version!=':latest-published':raise ValueError('Use a released numeric version or :latest-published.')
    url=SERVER+'/api/datasets/:persistentId/versions/'+version+'?'+urlencode({'persistentId':doi})
    opener=open_url or (lambda request:urlopen(request,timeout=45))
    with opener(Request(url,headers={'Accept':'application/json'})) as response:
        raw=response.read(32*1024**2+1)
    if len(raw)>32*1024**2:raise ValueError('Metadata exceeds 32 MiB.')
    catalog=parse_inventory(json.loads(raw),doi=doi);catalog['metadata_url']=url;return catalog


class SafeRedirect(HTTPRedirectHandler):
    def redirect_request(self,req,fp,code,msg,headers,newurl):
        if urlparse(newurl).scheme!='https':raise ValueError('Non-HTTPS redirect refused.')
        new=super().redirect_request(req,fp,code,msg,headers,newurl)
        if urlparse(newurl).hostname!=urlparse(SERVER).hostname:
            new.remove_header('X-dataverse-key')
        return new


def download_selected(inventory:dict,file_ids:list[int],out:Path,*,accept_noncommercial=False,
                      max_total_bytes=256*1024**2,open_url=None) -> list[dict]:
    if not accept_noncommercial:raise ValueError('Explicit CC BY-NC research-use acknowledgement required.')
    if not file_ids or len(set(file_ids))!=len(file_ids):raise ValueError('Select distinct file IDs from the inventory.')
    byid={x['file_id']:x for x in inventory['files']}
    if any(i not in byid for i in file_ids):raise ValueError('Selected ID is absent from inventory.')
    entries=[byid[i] for i in file_ids]
    if any(e['restricted'] for e in entries):raise ValueError('Restricted files are not downloaded by this helper. Complete the dataset access process separately.')
    if sum(e['bytes'] for e in entries)>max_total_bytes:raise ValueError('Selection exceeds explicit download budget.')
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    opener=open_url or (lambda req:build_opener(SafeRedirect()).open(req,timeout=60))
    results=[]
    for entry in entries:
        check=entry['checksum'];algo=str(check.get('type','')).lower().replace('-','')
        if algo not in ('md5','sha1','sha256','sha512') or not check.get('value'):raise ValueError('File lacks a supported repository checksum; refusing unverified download.')
        rel=safe_relative(entry['path']);dest=out/rel
        if out not in dest.resolve().parents:raise ValueError('Download destination escapes root.')
        dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():raise ValueError(f'Refusing to overwrite existing file: {rel}')
        temporary=dest.with_suffix(dest.suffix+'.part')
        if temporary.exists():raise ValueError(f'Partial file already exists: {rel}.part; inspect/remove before retrying.')
        url=SERVER+f"/api/access/datafile/{entry['file_id']}"
        req=Request(url)
        token=os.environ.get('DATAVERSE_API_TOKEN')
        if token:req.add_header('X-Dataverse-key',token)
        h=hashlib.new(algo);sha=hashlib.sha256();count=0
        try:
            with opener(req) as response,temporary.open('xb') as file:
                if response.headers.get('Content-Type','').split(';')[0] in ('text/html','application/json'):
                    raise ValueError('Received login/error document instead of mesh bytes.')
                while True:
                    chunk=response.read(1024**2)
                    if not chunk:break
                    count+=len(chunk)
                    if count>entry['bytes'] or count>max_total_bytes:raise ValueError('Download exceeds advertised size/budget.')
                    h.update(chunk);sha.update(chunk);file.write(chunk)
            if count!=entry['bytes'] or h.hexdigest().lower()!=check['value'].lower():raise ValueError('Download checksum or byte length mismatch.')
            temporary.replace(dest)
        except Exception:
            temporary.unlink(missing_ok=True);raise
        record=dict(file_id=entry['file_id'],path=rel,sha256=sha.hexdigest(),repository_checksum=check,
            bytes=count,dataset_doi=inventory['dataset_doi'],dataset_version=inventory['version'],verified=True,
            license='CC BY-NC 4.0',geometry_or_CFD_validated=False)
        (dest.with_suffix(dest.suffix+'.provenance.json')).write_text(json.dumps(record,indent=2),encoding='utf-8')
        results.append(record)
    return results
