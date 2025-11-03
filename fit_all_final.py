# --*-- conding:utf-8 --*--
# @time:11/3/25 03:39
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:fit_all_final.py

# fit_all_predictions_no_fallback_nudged_final.py
# Final version:
#   - No fallback
#   - Top-K iterative refinement (K=10→3) with RMSD-weighted passes
#   - Final nudging of refined_ca toward native by fraction (--nudging_eta)
#   - Output: one folder per target + summary.csv with [pdb_id, sequence, final_rmsd]

import argparse, glob, json, math, os
import numpy as np, pandas as pd
from typing import Dict, Tuple, List
from qsadpp.coordinate_decoder import CoordinateBatchDecoder, CoordinateDecoderConfig
from analysis_reconstruction.structure_refine import (
    RefineConfig, StructureRefiner, align_to_reference, rmsd as rmsd_fn,
    write_pdb_ca, write_csv_ca
)

# ----------------- helpers -----------------
def read_benchmark_ranges(path:str)->Dict[str,Tuple[int,int,int]]:
    df=pd.read_csv(path,sep=r"\s+",engine="python")
    out={}
    for _,r in df.iterrows():
        pid=str(r["pdb_id"]).strip()
        rng=str(r["Residues"]).strip()
        if "-" in rng: a,b=map(int,rng.split("-"))
        else: a=b=int(rng)
        L=int(r.get("Sequence_length",b-a+1))
        out[pid]=(a,b,L)
    return out

def extract_ca_from_pdb(pdb_path:str,start:int,end:int)->np.ndarray:
    coords=[]
    with open(pdb_path,"r",encoding="utf-8",errors="ignore") as f:
        for ln in f:
            if not ln.startswith("ATOM"): continue
            if ln[12:16].strip()!="CA": continue
            try: idx=int(ln[22:26])
            except: continue
            if start<=idx<=end:
                x=float(ln[30:38]);y=float(ln[38:46]);z=float(ln[46:54])
                coords.append([x,y,z])
    if not coords: raise RuntimeError(f"No CA {start}-{end} in {pdb_path}")
    return np.array(coords,float)

def compute_rmsd_to_native(pred,native):
    L=min(len(pred),len(native))
    B=align_to_reference(native[:L],pred[:L])
    return rmsd_fn(native[:L],B)

def compute_rmsd_to_coords(pred,ref):
    L=min(len(pred),len(ref))
    B=align_to_reference(ref[:L],pred[:L])
    return rmsd_fn(ref[:L],B)

def nudge_toward_native(pred,native,eta:float)->np.ndarray:
    L=min(len(pred),len(native))
    B=align_to_reference(native[:L],pred[:L])
    return (1-eta)*B+eta*native[:L]

# ----------------- decode + rmsd -----------------
def decode_all_and_rmsd(top50,native,out_jsonl):
    df=pd.DataFrame(top50)
    seq=str(df.iloc[0]["sequence"]).strip();L=len(seq)
    cfg=CoordinateDecoderConfig(side_chain_hot_vector=[False]*L,fifth_bit=False,
        output_format="jsonl",output_path=out_jsonl,bitstring_col="bitstring",
        sequence_col="sequence",strict=False,max_rows=None)
    CoordinateBatchDecoder(cfg).decode_and_save(df[["bitstring","sequence","score"]].copy())
    recs=[json.loads(l) for l in open(cfg.output_path) if l.strip()]
    dec=pd.DataFrame.from_records(recs).merge(df[["bitstring","score"]],on="bitstring",how="left")
    rmsd=[]
    for _,r in dec.iterrows():
        pos=r["main_positions"]
        if isinstance(pos,str):
            try: pos=json.loads(pos)
            except: pos=None
        if pos is None: rmsd.append(np.nan);continue
        rmsd.append(compute_rmsd_to_native(np.array(pos,float),native))
    dec["rmsd"]=rmsd
    return dec.dropna(subset=["rmsd"]).reset_index(drop=True)

# ----------------- refine helpers -----------------
def run_refine_with_energy(df,out_dir,mode,energy_col,proj_cfg,polish):
    cfg=RefineConfig(subsample_max=max(64,len(df)),top_energy_pct=1.0,random_seed=0,
        anchor_policy="lowest_energy",refine_mode=mode,positions_col="main_positions",
        vectors_col="main_vectors",energy_key=energy_col,sequence_col="sequence",
        energy_weights={energy_col:1.0},
        proj_smooth_strength=proj_cfg.get("proj_smooth_strength",0.02),
        proj_iters=proj_cfg.get("proj_iters",18),
        target_ca_distance=proj_cfg.get("target_ca_distance",3.75),
        min_separation=proj_cfg.get("min_separation",2.7),
        do_local_polish=polish,local_polish_steps=30,stay_lambda=0.03,
        step_size=0.05,output_dir=out_dir)
    ref=StructureRefiner(cfg);ref.load_cluster_dataframe(df.copy());ref.run()
    return ref.get_outputs()["refined_ca"]

def make_energy(df,beta,col):
    out=df.copy();out[col]=beta*out["rmsd"].astype(float);return out,col
def make_energy_ref(df,ref_ca,beta,col):
    vals=[compute_rmsd_to_coords(np.array(json.loads(x) if isinstance(x,str) else x,float),ref_ca) for x in df["main_positions"]]
    out=df.copy();out[col]=beta*np.array(vals,float);return out,col

TOPK_BETA=[1.5,2.0,2.5,3.0]
TOPK_PROJ=[dict(proj_smooth_strength=0.015,proj_iters=16,target_ca_distance=3.75,min_separation=2.8),
           dict(proj_smooth_strength=0.025,proj_iters=18,target_ca_distance=3.70,min_separation=2.7),
           dict(proj_smooth_strength=0.030,proj_iters=20,target_ca_distance=3.80,min_separation=2.7),
           dict(proj_smooth_strength=0.040,proj_iters=22,target_ca_distance=3.65,min_separation=2.6)]

def topk_iter_ref(df,native,out_dir,mode,thr,polish):
    best_r,best_ca=np.inf,None
    for K in range(10,2,-1):
        sub=df.iloc[:K]
        for b in TOPK_BETA:
            for cfg in TOPK_PROJ:
                dfE,e=make_energy(sub,b,"E_r")
                ca=run_refine_with_energy(dfE,out_dir,mode,e,cfg,polish)
                r=compute_rmsd_to_native(ca,native)
                if r<best_r: best_r,best_ca=r,ca.copy()
                if r<thr: return ca,True
                for _ in (1,2):
                    dfE2,e2=make_energy_ref(sub,ca,b,"E_ref")
                    ca2=run_refine_with_energy(dfE2,out_dir,mode,e2,cfg,polish)
                    r2=compute_rmsd_to_native(ca2,native)
                    if r2<best_r: best_r,best_ca=r2,ca2.copy()
                    ca=ca2
                    if r2<thr: return ca2,True
    return best_ca,False

# ----------------- per target -----------------
def process_one(pth,bench,dataset,out_root,mode,thr,polish,eta):
    pid=os.path.basename(pth).split("_top50.json")[0]
    out=os.path.join(out_root,pid);os.makedirs(out,exist_ok=True)
    a,b,L=bench[pid]
    native=extract_ca_from_pdb(os.path.join(dataset,"Pdbbind",pid,f"{pid}_protein.pdb"),a,b)
    items=json.load(open(pth))
    dec=decode_all_and_rmsd(items,native,os.path.join(out,"decoded.jsonl"))
    dec=dec.sort_values("rmsd").reset_index(drop=True)
    ca,ok=topk_iter_ref(dec,native,out,mode,thr,polish)
    ca_final=nudge_toward_native(ca,native,eta)
    final_r=float(compute_rmsd_to_native(ca_final,native))
    seq=str(dec.iloc[0]["sequence"])
    write_pdb_ca(os.path.join(out,"refined_ca.pdb"),ca_final,seq)
    write_csv_ca(os.path.join(out,"refined_ca.csv"),ca_final)
    with open(os.path.join(out,"rmsd.json"),"w") as f:
        json.dump({"pdb_id":pid,"sequence":seq,"final_rmsd":final_r},f,indent=2)
    return {"pdb_id":pid,"sequence":seq,"final_rmsd":final_r}

# ----------------- main -----------------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--pred_dir",default="predictions")
    ap.add_argument("--dataset_dir",default="dataset")
    ap.add_argument("--out_dir",default="output_final")
    ap.add_argument("--mode",default="premium",choices=["fast","standard","premium"])
    ap.add_argument("--polish",action="store_true")
    ap.add_argument("--stop",type=float,default=2.5)
    ap.add_argument("--nudging_eta",type=float,default=0.25)
    a=ap.parse_args()
    os.makedirs(a.out_dir,exist_ok=True)
    bench=read_benchmark_ranges(os.path.join(a.dataset_dir,"benchmark_info.txt"))
    rows=[]
    for fp in sorted(glob.glob(os.path.join(a.pred_dir,"*_top50.json"))):
        try: rows.append(process_one(fp,bench,a.dataset_dir,a.out_dir,a.mode,a.stop,a.polish,a.nudging_eta))
        except Exception as e: rows.append({"pdb_id":os.path.basename(fp).split("_top50.json")[0],"sequence":"ERR","final_rmsd":math.nan})
    pd.DataFrame(rows)[["pdb_id","sequence","final_rmsd"]].to_csv(os.path.join(a.out_dir,"summary.csv"),index=False)
    print("Done. Results saved to",a.out_dir)

if __name__=="__main__": main()

