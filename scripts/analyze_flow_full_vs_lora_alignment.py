#!/usr/bin/env python3
from __future__ import annotations

import argparse,csv,gc,math,re
from collections import defaultdict
from pathlib import Path
import torch

FLOW_PREFIX='policy_backend.flow.'
TARGET_PATTERNS={
'attn_q':re.compile(r'^DiT\.transformer_blocks\.(\d+)\.attn1\.to_q\.weight$'),
'attn_k':re.compile(r'^DiT\.transformer_blocks\.(\d+)\.attn1\.to_k\.weight$'),
'attn_v':re.compile(r'^DiT\.transformer_blocks\.(\d+)\.attn1\.to_v\.weight$'),
'attn_o':re.compile(r'^DiT\.transformer_blocks\.(\d+)\.attn1\.to_out\.0\.weight$'),
'ff_in':re.compile(r'^DiT\.transformer_blocks\.(\d+)\.ff\.net\.0\.proj\.weight$'),
'ff_out':re.compile(r'^DiT\.transformer_blocks\.(\d+)\.ff\.net\.2\.weight$'),
}

def load_state(path:Path):
    try: obj=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    except TypeError:
        try: obj=torch.load(path,map_location='cpu',weights_only=True)
        except TypeError: obj=torch.load(path,map_location='cpu')
    if isinstance(obj,dict):
        for w in ('state_dict','model','module'):
            n=obj.get(w)
            if isinstance(n,dict) and n and any(torch.is_tensor(v) for v in n.values()): obj=n; break
    if not isinstance(obj,dict): raise RuntimeError(f'Unsupported checkpoint: {path}')
    return obj

def flow_view(sd):
    out={str(k)[len(FLOW_PREFIX):]:v for k,v in sd.items() if torch.is_tensor(v) and str(k).startswith(FLOW_PREFIX)}
    if not out: raise RuntimeError('No policy_backend.flow.* tensors found')
    return out

def block_of(k):
    m=re.match(r'^DiT\.transformer_blocks\.(\d+)\.',k); return int(m.group(1)) if m else None

def classify(k):
    for name,pat in TARGET_PATTERNS.items():
        m=pat.match(k)
        if m:return name,int(m.group(1)),True
    b=block_of(k)
    if k.startswith('DiT.transformer_blocks.') and '.norm' in k:return 'transformer_norm',b,False
    if k.startswith('DiT.timestep_encoder.'):return 'timestep_encoder',None,False
    if k.startswith('DiT.proj_out_'):return 'output_projection',None,False
    if k.startswith('enc_vlm.'):return 'enc_vlm',None,False
    if k.startswith('enc_wm.'):return 'enc_wm',None,False
    if k.startswith('action_encoder.'):return 'action_encoder',None,False
    if k.startswith('action_decoder.'):return 'action_decoder',None,False
    if k.startswith('DiT.transformer_blocks.'):return 'transformer_other',b,False
    if k.startswith('DiT.'):return 'dit_other',None,False
    return 'flow_other',None,False

def write_csv(path,rows):
    if not rows:return
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys()));w.writeheader();w.writerows(rows)

@torch.no_grad()
def metrics(base_t,full_t,cand_t,device):
    b=base_t.detach().to(device=device,dtype=torch.float32)
    f=full_t.detach().to(device=device,dtype=torch.float32)
    c=cand_t.detach().to(device=device,dtype=torch.float32)
    df=f-b; dc=c-b; residual=dc-df
    ef=float((df*df).sum());ec=float((dc*dc).sum());er=float((residual*residual).sum());dot=float((df*dc).sum())
    nf=math.sqrt(max(ef,0));nc=math.sqrt(max(ec,0))
    out={
      'full_update_energy':ef,'candidate_update_energy':ec,'residual_energy':er,'dot':dot,
      'full_update_norm':nf,'candidate_update_norm':nc,
      'norm_ratio':nc/nf if nf>0 else float('nan'),
      'cosine_alignment':dot/(nf*nc) if nf>0 and nc>0 else float('nan'),
      'projection_coeff_on_full':dot/ef if ef>0 else float('nan'),
      'relative_residual':math.sqrt(er/ef) if ef>0 else float('nan'),
      'reconstruction_score':1-er/ef if ef>0 else float('nan')}
    del b,f,c,df,dc,residual
    return out

def fresh():return {'n':0,'params':0,'full':0.0,'cand':0.0,'resid':0.0,'dot':0.0}
def add(acc,m,n):
    acc['n']+=1;acc['params']+=n;acc['full']+=m['full_update_energy'];acc['cand']+=m['candidate_update_energy'];acc['resid']+=m['residual_energy'];acc['dot']+=m['dot']
def aggregate(a):
    ef,ec,er,dot=a['full'],a['cand'],a['resid'],a['dot']; nf=math.sqrt(max(ef,0));nc=math.sqrt(max(ec,0))
    return {'num_tensors':a['n'],'num_params':a['params'],'full_update_energy':ef,'candidate_update_energy':ec,
    'candidate_to_full_norm_ratio':nc/nf if nf>0 else float('nan'),
    'weighted_cosine_alignment':dot/(nf*nc) if nf>0 and nc>0 else float('nan'),
    'projection_coeff_on_full':dot/ef if ef>0 else float('nan'),
    'relative_residual':math.sqrt(er/ef) if ef>0 else float('nan'),
    'reconstruction_score':1-er/ef if ef>0 else float('nan')}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--base',required=True,type=Path);ap.add_argument('--full',required=True,type=Path)
    ap.add_argument('--candidate',action='append',nargs=2,metavar=('LABEL','CHECKPOINT'),required=True)
    ap.add_argument('--output-dir',required=True,type=Path);ap.add_argument('--device',default='cuda:0' if torch.cuda.is_available() else 'cpu')
    args=ap.parse_args();device=torch.device(args.device);args.output_dir.mkdir(parents=True,exist_ok=True)
    print('='*100);print('T9 Full-Flow vs Flow-LoRA Update Alignment');print('='*100)
    print(f'Base : {args.base}\nFull : {args.full}\nDevice: {device}\n')
    base_sd=load_state(args.base);full_sd=load_state(args.full);base=flow_view(base_sd);full=flow_view(full_sd)
    if set(base)!=set(full):raise RuntimeError('Base/Full canonical Flow key mismatch')
    total_full=target_full=0.0
    for key in sorted(base):
        b=base[key].detach().to(device=device,dtype=torch.float32);f=full[key].detach().to(device=device,dtype=torch.float32);e=float(((f-b)**2).sum())
        total_full+=e
        if classify(key)[2]:target_full+=e
        del b,f
    coverage=target_full/total_full if total_full>0 else 0.0
    print(f'Strict Full-Flow 96-target coverage: {100*coverage:.2f}%\n')
    tensor_rows=[];component_rows=[];block_rows=[];overview=[]
    for label,path in args.candidate:
        path=Path(path);print('-'*100);print(f'Candidate {label}: {path}')
        cand_sd=load_state(path);cand=flow_view(cand_sd)
        if set(cand)!=set(base):raise RuntimeError(f'{label}: canonical Flow key mismatch')
        global_acc=fresh();target_acc=fresh();nontarget_acc=fresh();components=defaultdict(fresh);blocks=defaultdict(fresh)
        for i,key in enumerate(sorted(base),1):
            cat,block,is_target=classify(key);m=metrics(base[key],full[key],cand[key],device)
            tensor_rows.append({'candidate':label,'key':key,'component':cat,'block':'' if block is None else block,'is_current_lora_target':int(is_target),'shape':'x'.join(map(str,base[key].shape)),'num_params':base[key].numel(),**m})
            add(global_acc,m,base[key].numel());add(target_acc if is_target else nontarget_acc,m,base[key].numel());add(components[cat],m,base[key].numel())
            if block is not None:add(blocks[block],m,base[key].numel())
            if i%50==0 or i==len(base):print(f'  alignment {i}/{len(base)}')
        g,t,nt=aggregate(global_acc),aggregate(target_acc),aggregate(nontarget_acc)
        overview.append({'candidate':label,'strict_full_target_coverage':coverage,
        'all_flow_weighted_cosine':g['weighted_cosine_alignment'],'all_flow_norm_ratio':g['candidate_to_full_norm_ratio'],'all_flow_projection_coeff':g['projection_coeff_on_full'],'all_flow_relative_residual':g['relative_residual'],'all_flow_reconstruction_score':g['reconstruction_score'],
        'target_only_weighted_cosine':t['weighted_cosine_alignment'],'target_only_norm_ratio':t['candidate_to_full_norm_ratio'],'target_only_projection_coeff':t['projection_coeff_on_full'],'target_only_relative_residual':t['relative_residual'],'target_only_reconstruction_score':t['reconstruction_score'],
        'nontarget_candidate_update_norm_ratio':nt['candidate_to_full_norm_ratio']})
        for comp,acc in sorted(components.items()):component_rows.append({'candidate':label,'component':comp,**aggregate(acc)})
        for block,acc in sorted(blocks.items()):block_rows.append({'candidate':label,'block':block,**aggregate(acc)})
        print(f"[{label}] all-flow cosine={g['weighted_cosine_alignment']:.4f}, norm_ratio={g['candidate_to_full_norm_ratio']:.4f}, relative_residual={g['relative_residual']:.4f}")
        print(f"[{label}] target-only cosine={t['weighted_cosine_alignment']:.4f}, norm_ratio={t['candidate_to_full_norm_ratio']:.4f}, relative_residual={t['relative_residual']:.4f}")
        del cand,cand_sd;gc.collect();
        if device.type=='cuda':torch.cuda.empty_cache()
    write_csv(args.output_dir/'alignment_overview.csv',overview);write_csv(args.output_dir/'per_tensor_alignment.csv',tensor_rows);write_csv(args.output_dir/'component_alignment.csv',component_rows);write_csv(args.output_dir/'block_alignment.csv',block_rows)
    with (args.output_dir/'SUMMARY.txt').open('w',encoding='utf-8') as f:
        f.write('T9 Strict Full-Flow vs Flow-LoRA Update Alignment\n'+'='*72+'\n\n');f.write(f'Base: {args.base}\nFull: {args.full}\nStrict Full-Flow current-96-target coverage: {100*coverage:.2f}%\n\n')
        for r in overview:
            f.write(f"{r['candidate']}\n  all Flow: cosine={r['all_flow_weighted_cosine']:.4f}, norm_ratio={r['all_flow_norm_ratio']:.4f}, projection={r['all_flow_projection_coeff']:.4f}, relative_residual={r['all_flow_relative_residual']:.4f}\n  targets : cosine={r['target_only_weighted_cosine']:.4f}, norm_ratio={r['target_only_norm_ratio']:.4f}, projection={r['target_only_projection_coeff']:.4f}, relative_residual={r['target_only_relative_residual']:.4f}\n\n")
    print('\n[OK] Saved:')
    for n in ('alignment_overview.csv','component_alignment.csv','block_alignment.csv','per_tensor_alignment.csv','SUMMARY.txt'):print(' ',args.output_dir/n)
if __name__=='__main__':main()