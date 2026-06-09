"""STEP 38 / pre-gate #2 — FAIR SSL check. Our 'parity with DER on real images' gives OURS a self-sup AE trunk DER
never gets. Test: does the same self-sup pretraining help a STANDARD CNN (DER's backbone) too? Cheap single-task
pre-check: SSL-init vs random-init SimpleCNN single-task accuracy on Tetro red-shape. If SSL-init >= random-init
(helps/neutral, expected since DER finetunes end-to-end), then SSL-DER >= DER -> our 'beats/parity DER' on real
images is an unmatched-pretraining artifact -> retire the claim. Only if SSL gives NEGATIVE transfer could the fair
test favor us.
Usage: python step38_ssl_der.py --seeds 0 1 2 3 --device cuda
"""
import sys, os, json, argparse
sys.path.insert(0, "/tmp"); sys.path.insert(0, "experiments/m2"); sys.path.insert(0, "experiments/m1_wk0")
import numpy as np, torch, torch.nn as nn
import step31_tetrominoes_task as s31
import m2_shapes_construct as shp

def _conv_stack():
    return nn.Sequential(nn.Conv2d(3,32,5,2,2),nn.ReLU(),nn.Conv2d(32,64,3,2,1),nn.ReLU(),nn.Conv2d(64,64,3,1,1),nn.ReLU())
class CNN(nn.Module):
    def __init__(self, nc=6):
        super().__init__(); self.f=_conv_stack(); self.pool=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Flatten()); self.c=nn.Linear(64,nc)
    def forward(self,x): return self.c(self.pool(self.f(x)))

def pretrain_ae(seed, device, n_per_class=300, epochs=15):
    X,_ = shp._gen_split(n_per_class, seed=2000+seed); Xt=torch.tensor(X,dtype=torch.float32); torch.manual_seed(40_000+seed)
    enc=_conv_stack().to(device)
    dec=nn.Sequential(nn.ConvTranspose2d(64,64,3,1,1),nn.ReLU(),nn.ConvTranspose2d(64,32,3,2,1,output_padding=1),nn.ReLU(),
                      nn.ConvTranspose2d(32,3,5,2,2,output_padding=1),nn.Sigmoid()).to(device)
    opt=torch.optim.Adam(list(enc.parameters())+list(dec.parameters()),lr=1e-3); mse=nn.MSELoss(); enc.train(); dec.train()
    for ep in range(epochs):
        pr=torch.randperm(len(Xt))
        for i in range(0,len(Xt),128):
            xb=Xt[pr[i:i+128]].to(device); opt.zero_grad(); loss=mse(dec(enc(xb)),xb); loss.backward(); opt.step()
    return enc

def single_task_acc(seed, ssl, device, epochs=40):
    # task 0 = classes 0,1 of the Tetro red-shape stream
    Xtr,ytr = s31._gen_split(600, seed=1000+seed); Xte,yte = s31._gen_split(200, seed=5000+seed)
    tr = s31._experiences(Xtr,ytr,n_exp=3)[0]; te = s31._experiences(Xte,yte,n_exp=3)[0]
    Xc=torch.tensor(tr[0],dtype=torch.float32); yc=torch.tensor(tr[1],dtype=torch.long).to(device)
    Xk=torch.tensor(te[0],dtype=torch.float32); yk=torch.tensor(te[1],dtype=torch.long).to(device)
    torch.manual_seed(seed); model=CNN(6).to(device)
    if ssl:
        enc=pretrain_ae(seed,device); model.f.load_state_dict(enc.state_dict())
    opt=torch.optim.Adam(model.parameters(),lr=1e-3); crit=nn.CrossEntropyLoss()
    for ep in range(epochs):
        pr=torch.randperm(len(yc))
        for i in range(0,len(yc),128):
            idx=pr[i:i+128]; opt.zero_grad(); loss=crit(model(Xc[idx].to(device)),yc[idx]); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        acc=float((model(Xk.to(device)).argmax(1)==yk).float().mean())
    return acc

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--seeds",type=int,nargs="+",default=[0,1,2,3]); ap.add_argument("--device",default="cuda"); ap.add_argument("--out",default="step38_ssl_der.json")
    a=ap.parse_args(); path="experiments/m2/results/"+a.out; recs=[]
    for s in a.seeds:
        rnd=single_task_acc(s,False,a.device); ssl=single_task_acc(s,True,a.device); recs.append({"seed":s,"random_init":rnd,"ssl_init":ssl})
        json.dump({"runs":recs},open(path,"w"),indent=2,default=str)
        print(f"[sslDER s{s}] random_init={rnd:.3f}  ssl_init={ssl:.3f}  delta={ssl-rnd:+.3f}  (SSL>=random => SSL helps DER => retire our edge)",flush=True)
    print("STEP38_DONE")
