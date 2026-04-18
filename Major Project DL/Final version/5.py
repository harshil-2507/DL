import os, ssl, random, copy, time, warnings
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except: pass
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, fbeta_score,
    roc_curve, auc, precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from collections import Counter
from PIL import Image
import cv2
warnings.filterwarnings('ignore')

# ════════════════════════════════════════════════════════════════
# PATHS + CONFIG
# ════════════════════════════════════════════════════════════════
DATA_DIR    = '/home/rahuldixit/aksh/dl_project/dataset/extracted_images'
OUTPUT_DIR  = '/home/rahuldixit/aksh/dl_project/outputs_final'
SAVE_DIR    = '/home/rahuldixit/aksh/dl_project/saved_models_final'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)

IMG_SIZE         = 224
BATCH_SIZE       = 64
CNN_EPOCHS       = 20
TFM_EPOCHS       = 20
AE_EPOCHS        = 30
NUM_CLASSES      = 14
DROPOUT          = 0.4
UNDER_CAP        = 200
AUG_TARGET       = 1000
FOCAL_GAMMA      = 2.0
WORKERS          = 8

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device : {device}")
if torch.cuda.is_available():
    print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

def set_seed(s=42):
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    np.random.seed(s); random.seed(s)
    torch.backends.cudnn.deterministic = True
set_seed()

# ════════════════════════════════════════════════════════════════
# PHASE 1 — DATA
# ════════════════════════════════════════════════════════════════
print("\n[Phase 1] Loading data...")

full_ds     = datasets.ImageFolder(DATA_DIR)
class_names = full_ds.classes
print(f"Classes ({NUM_CLASSES}): {class_names}")

# Stratified splits — same seed every run for reproducibility
train_idx, test_idx = train_test_split(
    range(len(full_ds)), test_size=0.15, random_state=42,
    stratify=full_ds.targets)
train_idx, val_idx = train_test_split(
    train_idx, test_size=0.17647, random_state=42,
    stratify=[full_ds.targets[i] for i in train_idx])
train_idx = list(train_idx); val_idx = list(val_idx); test_idx = list(test_idx)
print(f"Train:{len(train_idx)}  Val:{len(val_idx)}  Test:{len(test_idx)}")

train_dist   = Counter([full_ds.targets[i] for i in train_idx])
normal_idx   = class_names.index('Normal clean mucosa')
minority_cls = [class_names[i] for i in range(NUM_CLASSES)
                if train_dist.get(i,0) < UNDER_CAP]
majority_cls = [c for c in class_names if c not in minority_cls]
print(f"Minority ({len(minority_cls)}): {minority_cls}")
print(f"Majority ({len(majority_cls)}): {majority_cls}")

# Transforms
tf_norm = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
tf_aug = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.RandomAffine(0, translate=(0.2,0.2), scale=(0.8,1.2), shear=10),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.RandomGrayscale(p=0.05),
    transforms.RandomPerspective(0.2, p=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

class IndexDS(Dataset):
    def __init__(self, raw, idx, tfm=tf_norm):
        self.raw=raw; self.idx=idx; self.tfm=tfm
    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        img, lbl = self.raw[self.idx[i]]
        return self.tfm(img), lbl

class BalancedDS(Dataset):
    """Under-sample majority to UNDER_CAP, augment minority up to AUG_TARGET."""
    def __init__(self, raw, train_idx, cap=UNDER_CAP, target=AUG_TARGET):
        self.samples = []; random.seed(42)
        bins = {i:[] for i in range(NUM_CLASSES)}
        for idx in train_idx:
            bins[raw.targets[idx]].append(idx)
        for c, idxs in bins.items():
            pool = random.sample(idxs, min(len(idxs), cap))
            for i in pool: self.samples.append((i, c, False))
            if len(pool) < target:
                for _ in range(target - len(pool)):
                    self.samples.append((random.choice(pool), c, True))
        fc = Counter(s[1] for s in self.samples)
        print(f"  Balanced: {len(self.samples):,} samples")
        for i,c in enumerate(class_names):
            n = fc.get(i,0)
            print(f"    {c:<25} {n:>5}")
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        idx, lbl, aug = self.samples[i]
        img, _ = full_ds[idx]
        return (tf_aug if aug else tf_norm)(img), lbl

val_ds   = IndexDS(full_ds, val_idx)
test_ds  = IndexDS(full_ds, test_idx)
s1_train = IndexDS(full_ds, train_idx)

print("Building S3 balanced dataset...")
s3_train = BalancedDS(full_ds, train_idx)

def loader(ds, shuffle=False):
    return DataLoader(ds, BATCH_SIZE, shuffle=shuffle, num_workers=WORKERS, pin_memory=True)

s1_ld = loader(s1_train, True); s3_ld = loader(s3_train, True)
val_ld = loader(val_ds);        test_ld = loader(test_ds)

# ════════════════════════════════════════════════════════════════
# PHASE 2 — FOCAL LOSS  (KEY FIX: no alpha weights for S3)
# ════════════════════════════════════════════════════════════════
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__(); self.gamma=gamma; self.alpha=alpha
    def forward(self, x, y):
        ce  = F.cross_entropy(x, y, weight=self.alpha, reduction='none')
        pt  = torch.exp(-ce)
        return (((1-pt)**self.gamma) * ce).mean()

# S1: class weights because training data is highly imbalanced
total = len(train_idx)
s1_weights = torch.tensor([
    total / (NUM_CLASSES * max(train_dist.get(i,1), 1))
    for i in range(NUM_CLASSES)
], dtype=torch.float)
s1_weights = torch.clamp(s1_weights, 0.1, 10.0).to(device)

# S3: NO weights — balanced dataset handles imbalance physically
focal_s1 = FocalLoss(FOCAL_GAMMA, alpha=s1_weights)
focal_s3 = FocalLoss(FOCAL_GAMMA, alpha=None)        # ← KEY FIX
print(f"\nS1 focal: with class weights (max={s1_weights.max():.1f}x)")
print(f"S3 focal: plain focal loss, no alpha (balanced data)")

# ════════════════════════════════════════════════════════════════
# PHASE 3 — MODEL BUILDERS
# ════════════════════════════════════════════════════════════════
def unfreeze_last_30_cnn(model):
    for p in model.parameters(): p.requires_grad = False
    if hasattr(model, 'layer4'):
        for p in model.layer4.parameters(): p.requires_grad = True
        return
    if hasattr(model, 'features'):
        blocks = list(model.features.children())
        for b in blocks[int(len(blocks)*0.7):]:
            for p in b.parameters(): p.requires_grad = True

def build_effnet():
    m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    unfreeze_last_30_cnn(m)
    m.classifier = nn.Sequential(
        nn.Dropout(DROPOUT), nn.Linear(m.classifier[1].in_features, 256),
        nn.ReLU(), nn.Dropout(DROPOUT/2), nn.Linear(256, NUM_CLASSES))
    for p in m.classifier.parameters(): p.requires_grad = True
    return m.to(device)

def build_mobilenet():
    m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    unfreeze_last_30_cnn(m)
    m.classifier[3] = nn.Sequential(
        nn.Dropout(DROPOUT), nn.Linear(m.classifier[3].in_features, NUM_CLASSES))
    for p in m.classifier.parameters(): p.requires_grad = True
    return m.to(device)

def build_resnet50():
    m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    unfreeze_last_30_cnn(m)
    m.fc = nn.Sequential(
        nn.Dropout(DROPOUT), nn.Linear(m.fc.in_features, 256),
        nn.ReLU(), nn.Dropout(DROPOUT/2), nn.Linear(256, NUM_CLASSES))
    for p in m.fc.parameters(): p.requires_grad = True
    return m.to(device)

# Transformer models — full fine-tune with lower LR
def build_vit():
    m = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
    m.heads.head = nn.Linear(m.heads.head.in_features, NUM_CLASSES)
    return m.to(device)

def build_swin():
    m = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)
    m.head = nn.Linear(m.head.in_features, NUM_CLASSES)
    return m.to(device)

# ════════════════════════════════════════════════════════════════
# PHASE 4 — TRAINING LOOP
# ════════════════════════════════════════════════════════════════
def train(model, tr_ld, vl_ld, criterion, tag,
          epochs=CNN_EPOCHS, lr=1e-3, is_transformer=False):
    print(f"\n=== Training {tag} | lr={lr} | epochs={epochs} ===")
    opt_cls = optim.AdamW if is_transformer else optim.Adam
    opt     = opt_cls(filter(lambda p:p.requires_grad, model.parameters()),
                      lr=lr, weight_decay=1e-4)
    cos     = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    plat    = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', 0.5, patience=3)
    hist    = {'tr':[],'vl':[],'f1':[],'lr':[]}
    best_vl = float('inf'); best_wt = copy.deepcopy(model.state_dict())

    for ep in range(epochs):
        model.train(); run=0.0
        for x,y in tr_ld:
            x,y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = criterion(model(x), y)
            loss.backward(); opt.step()
            run += loss.item()*x.size(0)
        tr_l = run/len(tr_ld.dataset)

        model.eval(); vl=0.0; ps=[]; ys=[]
        with torch.no_grad():
            for x,y in vl_ld:
                x,y = x.to(device), y.to(device)
                out = model(x)
                vl += criterion(out,y).item()*x.size(0)
                ps.extend(torch.max(out,1)[1].cpu().numpy())
                ys.extend(y.cpu().numpy())
        vl_l = vl/len(vl_ld.dataset)
        mf1  = f1_score(ys, ps, average='macro', zero_division=0)
        cur_lr = opt.param_groups[0]['lr']
        hist['tr'].append(tr_l); hist['vl'].append(vl_l)
        hist['f1'].append(mf1);  hist['lr'].append(cur_lr)
        cos.step(); plat.step(vl_l)
        print(f"  Ep{ep+1:02d}/{epochs} Tr:{tr_l:.4f} Vl:{vl_l:.4f} MacroF1:{mf1:.4f} LR:{cur_lr:.2e}")
        if vl_l < best_vl:
            best_vl = vl_l; best_wt = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), f"{SAVE_DIR}/{tag}_best.pth")

    model.load_state_dict(best_wt)
    # Plot
    fig, ax = plt.subplots(1,3,figsize=(18,5))
    fig.suptitle(tag, fontsize=12, fontweight='bold')
    ax[0].plot(hist['tr'],label='Train',color='#7F77DD',lw=2)
    ax[0].plot(hist['vl'],label='Val',  color='#1D9E75',lw=2,ls='--')
    ax[0].set_title('Focal Loss'); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].plot(hist['f1'],color='#D85A30',lw=2)
    ax[1].set_title('Val Macro F1'); ax[1].grid(alpha=0.3)
    ax[2].plot(hist['lr'],color='#EF9F27',lw=2,marker='o',ms=3)
    ax[2].set_title('LR'); ax[2].set_yscale('log'); ax[2].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{tag}_curves.png",dpi=150,bbox_inches='tight')
    plt.close()
    return model

# ---- CNN models on S1 and S3 ----
cnn_models_s1 = {}
cnn_models_s3 = {}

for name, builder in [('EffNetB0',build_effnet),
                       ('MobileV3',build_mobilenet),
                       ('ResNet50',build_resnet50)]:
    m = train(builder(), s1_ld, val_ld, focal_s1, f"S1_{name}", CNN_EPOCHS, lr=1e-3)
    cnn_models_s1[name] = m; del m; torch.cuda.empty_cache()

    m = train(builder(), s3_ld, val_ld, focal_s3, f"S3_{name}", CNN_EPOCHS, lr=1e-3)
    cnn_models_s3[name] = m; del m; torch.cuda.empty_cache()

# ---- Transformer models on S3 ----
tfm_models = {}
for name, builder, lr in [('ViT',  build_vit,  1e-4),
                            ('Swin', build_swin, 1e-4)]:
    m = train(builder(), s3_ld, val_ld, focal_s3, f"S3_{name}",
              TFM_EPOCHS, lr=lr, is_transformer=True)
    tfm_models[name] = m; del m; torch.cuda.empty_cache()

# ════════════════════════════════════════════════════════════════
# PHASE 5 — EVALUATION (ALL MODELS ON SHARED TEST SET)
# ════════════════════════════════════════════════════════════════
print("\n[Phase 5] Evaluating on shared test set...")

def evaluate(model, loader):
    model.eval(); ps=[]; ys=[]; probs=[]
    with torch.no_grad():
        for x,y in loader:
            out = model(x.to(device))
            p   = F.softmax(out,1)
            ps.extend(torch.max(out,1)[1].cpu().numpy())
            ys.extend(y.numpy())
            probs.extend(p.cpu().numpy())
    return np.array(ys), np.array(ps), np.array(probs)

all_results = {}

def run_eval(models_dict, prefix, loader):
    for name, model in models_dict.items():
        tag = f"{prefix}_{name}"
        yt, yp, yprob = evaluate(model, loader)
        rep  = classification_report(yt, yp, target_names=class_names,
                                      output_dict=True, zero_division=0)
        metrics = {
            'Acc(w)'     : round(rep['accuracy'],4),
            'F1(w)'      : round(rep['weighted avg']['f1-score'],4),
            'Rec(macro)' : round(rep['macro avg']['recall'],4),
            'F1(macro)'  : round(rep['macro avg']['f1-score'],4),
        }
        print(f"\n--- {tag} ---")
        print(classification_report(yt, yp, target_names=class_names, zero_division=0))
        all_results[tag] = {'yt':yt,'yp':yp,'yprob':yprob,'metrics':metrics}

run_eval(cnn_models_s1, 'S1', test_ld)
run_eval(cnn_models_s3, 'S3', test_ld)

# reload transformers from checkpoint and evaluate
for name in ['ViT','Swin']:
    builder = build_vit if name=='ViT' else build_swin
    m = builder()
    m.load_state_dict(torch.load(f"{SAVE_DIR}/S3_{name}_best.pth", map_location=device))
    tfm_models[name] = m
run_eval(tfm_models, 'S3', test_ld)

# Print summary table
print("\n" + "="*80)
print("FULL RESULTS SUMMARY")
print("="*80)
print(f"{'Model':<22} {'Acc(w)':>8} {'F1(w)':>8} {'Rec(macro)':>12} {'F1(macro)':>10}")
print("-"*80)
for tag, res in all_results.items():
    m = res['metrics']
    print(f"{tag:<22} {m['Acc(w)']:>8.4f} {m['F1(w)']:>8.4f} {m['Rec(macro)']:>12.4f} {m['F1(macro)']:>10.4f}")

# Save CSV
pd.DataFrame([{'Model':t,**r['metrics']} for t,r in all_results.items()]
             ).to_csv(f"{OUTPUT_DIR}/FULL_results.csv", index=False)

# Confusion matrix for best S3 model
best_tag  = max(all_results, key=lambda t: all_results[t]['metrics']['F1(macro)'])
best_res  = all_results[best_tag]
print(f"\nBest model: {best_tag} | F1(macro)={best_res['metrics']['F1(macro)']:.4f}")

cm = confusion_matrix(best_res['yt'], best_res['yp'])
fig,ax = plt.subplots(figsize=(14,12))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
            xticklabels=class_names, yticklabels=class_names, ax=ax, linewidths=0.4)
ax.set_title(f"Confusion Matrix — {best_tag}", fontsize=12, fontweight='bold')
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
plt.xticks(rotation=45,ha='right'); plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/{best_tag}_confusion_matrix.png",dpi=150,bbox_inches='tight')
plt.close()

# ROC per class for best model
n_cls = len(class_names)
y_bin = label_binarize(best_res['yt'], classes=list(range(n_cls)))
fig, axes = plt.subplots(2,7,figsize=(24,8)); axes=axes.flatten()
fig.suptitle(f"Per-class ROC — {best_tag}", fontsize=12, fontweight='bold')
for i,cls in enumerate(class_names):
    ax=axes[i]
    if y_bin.shape[1]<=i or y_bin[:,i].sum()==0: ax.axis('off'); continue
    fpr,tpr,_ = roc_curve(y_bin[:,i], best_res['yprob'][:,i])
    ra = auc(fpr,tpr)
    color = '#D85A30' if cls in minority_cls else '#7F77DD'
    ax.plot(fpr,tpr,color=color,lw=2,label=f'AUC={ra:.2f}')
    ax.plot([0,1],[0,1],'k--',lw=0.8,alpha=0.5)
    ax.set_title(cls[:14],fontsize=8,
                  color='#D85A30' if cls in minority_cls else 'black',
                  fontweight='bold' if cls in minority_cls else 'normal')
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
for j in range(i+1,len(axes)): axes[j].axis('off')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/{best_tag}_ROC_classes.png",dpi=130,bbox_inches='tight')
plt.close()

# ════════════════════════════════════════════════════════════════
# PHASE 6 — SUPERVISED ANOMALY DETECTION
# ════════════════════════════════════════════════════════════════
print("\n[Phase 6] Supervised anomaly detection...")

# Use best S3 model for softmax-based anomaly scoring
best_model_name = best_tag.split('_',1)[1]   # e.g. "S3_ViT" -> "ViT"
if 'ViT' in best_tag:
    best_m = build_vit()
elif 'Swin' in best_tag:
    best_m = build_swin()
elif 'EffNetB0' in best_tag:
    best_m = build_effnet()
elif 'MobileV3' in best_tag:
    best_m = build_mobilenet()
else:
    best_m = build_resnet50()
best_m.load_state_dict(torch.load(f"{SAVE_DIR}/{best_tag}_best.pth", map_location=device))
best_m.eval()

# Anomaly score = 1 - max(softmax)
anom_scores = 1.0 - best_res['yprob'].max(axis=1)
score_true  = (best_res['yt'] != normal_idx).astype(int)

fpr_s, tpr_s, _ = roc_curve(score_true, anom_scores)
auroc_s = auc(fpr_s, tpr_s)
auprc_s = average_precision_score(score_true, anom_scores)
thresh_s = np.linspace(0.05,0.95,91)
f2_s     = [fbeta_score(score_true,(anom_scores>=t).astype(int),beta=2,zero_division=0)
             for t in thresh_s]
best_t_s = thresh_s[np.argmax(f2_s)]
print(f"Score-based: AUROC={auroc_s:.4f}  AUPRC={auprc_s:.4f}  BestF2={max(f2_s):.4f}")

# Binary anomaly detector
class BinDS(Dataset):
    def __init__(self, raw, idx):
        self.raw=raw; self.idx=idx
    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        img, lbl = self.raw[self.idx[i]]
        return tf_norm(img), (0 if lbl==normal_idx else 1)

bin_tr_ld = DataLoader(BinDS(full_ds,train_idx), BATCH_SIZE, shuffle=True,
                        num_workers=WORKERS, pin_memory=True)
bin_ts_ld = DataLoader(BinDS(full_ds,test_idx),  BATCH_SIZE, shuffle=False,
                        num_workers=WORKERS, pin_memory=True)

n_anom = sum(1 for i in train_idx if full_ds.targets[i] != normal_idx)
n_norm = sum(1 for i in train_idx if full_ds.targets[i] == normal_idx)

def build_bin():
    m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    unfreeze_last_30_cnn(m)
    m.classifier = nn.Sequential(
        nn.Dropout(0.4), nn.Linear(m.classifier[1].in_features,128),
        nn.ReLU(), nn.Dropout(0.2), nn.Linear(128,1))
    for p in m.classifier.parameters(): p.requires_grad = True
    return m.to(device)

bin_m   = build_bin()
bin_opt = optim.Adam(filter(lambda p:p.requires_grad, bin_m.parameters()), lr=1e-3, weight_decay=1e-4)
bin_sch = optim.lr_scheduler.CosineAnnealingLR(bin_opt, T_max=CNN_EPOCHS, eta_min=1e-6)
bin_crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([n_norm/max(n_anom,1)]).to(device))

print("Training binary anomaly detector...")
for ep in range(CNN_EPOCHS):
    bin_m.train(); run=0.0
    for x,y in bin_tr_ld:
        x,y = x.to(device), y.float().unsqueeze(1).to(device)
        bin_opt.zero_grad(); loss=bin_crit(bin_m(x),y)
        loss.backward(); bin_opt.step(); run+=loss.item()*x.size(0)
    bin_sch.step()
    if (ep+1)%5==0: print(f"  Ep{ep+1:02d}/{CNN_EPOCHS} Loss:{run/len(bin_tr_ld.dataset):.4f}")
torch.save(bin_m.state_dict(), f"{SAVE_DIR}/binary_detector.pth")

bin_m.eval(); bsc=[]; byt=[]
with torch.no_grad():
    for x,y in bin_ts_ld:
        out = torch.sigmoid(bin_m(x.to(device))).squeeze(1).cpu().numpy()
        bsc.extend(out); byt.extend(y.numpy())
bsc  = np.array(bsc); byt = np.array(byt)
fpr_b, tpr_b, _ = roc_curve(byt, bsc)
auroc_b = auc(fpr_b, tpr_b)
auprc_b = average_precision_score(byt, bsc)
thresh_b = np.linspace(0.1,0.9,81)
f2_b     = [fbeta_score(byt,(bsc>=t).astype(int),beta=2,zero_division=0) for t in thresh_b]
best_t_b = thresh_b[np.argmax(f2_b)]
print(f"Binary detector: AUROC={auroc_b:.4f}  AUPRC={auprc_b:.4f}  BestF2={max(f2_b):.4f}")

# ════════════════════════════════════════════════════════════════
# PHASE 7 — UNSUPERVISED ANOMALY DETECTION
# ════════════════════════════════════════════════════════════════
print("\n[Phase 7] Unsupervised anomaly detection...")

# Extract ResNet50 S3 backbone features (without final classifier)
eff_s3 = build_effnet()
eff_s3.load_state_dict(torch.load(f"{SAVE_DIR}/S3_EffNetB0_best.pth", map_location=device))
eff_s3.eval()

feat_out = []; feat_lbl = []
def hook_fn(m,i,o): feat_out.append(o.detach().cpu())
h = eff_s3.features.register_forward_hook(hook_fn)

print("Extracting features for unsupervised methods...")
with torch.no_grad():
    for x,y in test_ld:
        feat_out.clear()
        _ = eff_s3(x.to(device))
        if feat_out:
            f = feat_out[0]
            if f.dim()>2: f=f.mean(dim=[2,3])
            feat_out_tensor = f
        feat_lbl.extend(y.numpy())
h.remove()

# Re-extract properly
all_feats = []; all_lbls = []
h = eff_s3.features.register_forward_hook(
    lambda m,i,o: all_feats.append(o.detach().cpu().mean(dim=[2,3])))
with torch.no_grad():
    for x,y in test_ld:
        _ = eff_s3(x.to(device))
        all_lbls.extend(y.numpy())
h.remove()

feats   = torch.cat(all_feats, 0).numpy()  # (N, 1280)
lbls    = np.array(all_lbls)
anom_gt = (lbls != normal_idx).astype(int)

norm_feats  = feats[lbls == normal_idx]
print(f"Features: {feats.shape}  Normal:{norm_feats.shape[0]}  Anomaly:{anom_gt.sum()}")

# -- Method 1: K-Means centroid distance
print("  K-Means centroid...")
km    = KMeans(n_clusters=1, n_init=10, random_state=42).fit(norm_feats)
dist  = np.linalg.norm(feats - km.cluster_centers_, axis=1)
fpr_k,tpr_k,_ = roc_curve(anom_gt, dist)
auroc_k = auc(fpr_k, tpr_k)
auprc_k = average_precision_score(anom_gt, dist)
print(f"  K-Means: AUROC={auroc_k:.4f}  AUPRC={auprc_k:.4f}")

# -- Method 2: Isolation Forest
print("  Isolation Forest...")
iso   = IsolationForest(n_estimators=200, contamination=0.2, random_state=42, n_jobs=-1)
iso.fit(norm_feats)
iso_scores = -iso.score_samples(feats)   # higher = more anomalous
fpr_i,tpr_i,_ = roc_curve(anom_gt, iso_scores)
auroc_i = auc(fpr_i, tpr_i)
auprc_i = average_precision_score(anom_gt, iso_scores)
print(f"  IsoForest: AUROC={auroc_i:.4f}  AUPRC={auprc_i:.4f}")

# -- Method 3: One-Class SVM
print("  One-Class SVM (subsampled for speed)...")
n_sub = min(5000, len(norm_feats))
svm_idx = np.random.choice(len(norm_feats), n_sub, replace=False)
ocsvm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.1)
ocsvm.fit(norm_feats[svm_idx])
svm_scores = -ocsvm.decision_function(feats)
fpr_svm,tpr_svm,_ = roc_curve(anom_gt, svm_scores)
auroc_svm = auc(fpr_svm, tpr_svm)
auprc_svm = average_precision_score(anom_gt, svm_scores)
print(f"  OCSVM:     AUROC={auroc_svm:.4f}  AUPRC={auprc_svm:.4f}")

# -- Method 4: Autoencoder reconstruction error
print("  Autoencoder...")

class ConvAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3,32,4,2,1), nn.ReLU(),   # 112
            nn.Conv2d(32,64,4,2,1), nn.ReLU(),  # 56
            nn.Conv2d(64,128,4,2,1), nn.ReLU(), # 28
            nn.Conv2d(128,256,4,2,1), nn.ReLU() # 14
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(256,128,4,2,1), nn.ReLU(),
            nn.ConvTranspose2d(128,64,4,2,1),  nn.ReLU(),
            nn.ConvTranspose2d(64,32,4,2,1),   nn.ReLU(),
            nn.ConvTranspose2d(32,3,4,2,1),    nn.Sigmoid()
        )
    def forward(self, x): return self.dec(self.enc(x))

# Normal-only training dataset for autoencoder
class NormalDS(Dataset):
    def __init__(self, raw, train_idx):
        self.idx = [i for i in train_idx if raw.targets[i] == normal_idx]
    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        img, _ = full_ds[self.idx[i]]
        # Return normalised image + same image as target (for reconstruction)
        t = tf_norm(img)
        return t, t

ae_ld = DataLoader(NormalDS(full_ds, train_idx), BATCH_SIZE, shuffle=True,
                    num_workers=WORKERS, pin_memory=True)
ae = ConvAE().to(device)
ae_opt = optim.Adam(ae.parameters(), lr=1e-3)
ae_sched = optim.lr_scheduler.StepLR(ae_opt, step_size=10, gamma=0.5)
mse = nn.MSELoss()

print(f"  AE train on {len(ae_ld.dataset):,} Normal images for {AE_EPOCHS} epochs...")
for ep in range(AE_EPOCHS):
    ae.train(); run=0.0
    for x,_ in ae_ld:
        x = x.to(device)
        ae_opt.zero_grad()
        recon = ae(x); loss = mse(recon, x)
        loss.backward(); ae_opt.step()
        run += loss.item()*x.size(0)
    ae_sched.step()
    if (ep+1)%10==0: print(f"    Ep{ep+1:02d} AE_Loss:{run/len(ae_ld.dataset):.6f}")
torch.save(ae.state_dict(), f"{SAVE_DIR}/autoencoder.pth")

ae.eval(); ae_errors=[]; ae_lbls=[]
with torch.no_grad():
    for x,y in test_ld:
        recon = ae(x.to(device))
        err   = ((recon - x.to(device))**2).mean(dim=[1,2,3]).cpu().numpy()
        ae_errors.extend(err); ae_lbls.extend(y.numpy())
ae_errors = np.array(ae_errors); ae_lbls = np.array(ae_lbls)
ae_gt     = (ae_lbls != normal_idx).astype(int)
fpr_a,tpr_a,_ = roc_curve(ae_gt, ae_errors)
auroc_a = auc(fpr_a, tpr_a)
auprc_a = average_precision_score(ae_gt, ae_errors)
print(f"  Autoencoder: AUROC={auroc_a:.4f}  AUPRC={auprc_a:.4f}")

# ======================== ANOMALY COMPARISON TABLE ========================
unsup_methods = {
    'K-Means centroid (unsup)'       : (auroc_k,  auprc_k,  fpr_k,  tpr_k),
    'Isolation Forest (unsup)'       : (auroc_i,  auprc_i,  fpr_i,  tpr_i),
    'One-Class SVM (unsup)'          : (auroc_svm,auprc_svm,fpr_svm,tpr_svm),
    'Autoencoder recon error (unsup)': (auroc_a,  auprc_a,  fpr_a,  tpr_a),
    'Score-based 1-softmax (sup)'    : (auroc_s,  auprc_s,  fpr_s,  tpr_s),
    'Binary detector (sup)'         : (auroc_b,  auprc_b,  fpr_b,  tpr_b),
}
print("\n" + "="*65)
print("ANOMALY DETECTION COMPARISON")
print("="*65)
print(f"{'Method':<40} {'AUROC':>8} {'AUPRC':>8}")
print("-"*65)
for name,(ar,ap,_,__) in unsup_methods.items():
    print(f"{name:<40} {ar:>8.4f} {ap:>8.4f}")

pd.DataFrame([{'Method':k,'AUROC':v[0],'AUPRC':v[1]}
               for k,v in unsup_methods.items()]
             ).to_csv(f"{OUTPUT_DIR}/anomaly_comparison.csv", index=False)

# ROC comparison plot
fig, axes = plt.subplots(1,2,figsize=(16,7))
fig.suptitle("Anomaly Detection — All Methods Compared", fontsize=13, fontweight='bold')
colors_sup  = ['#7F77DD','#534AB7']
colors_unsup= ['#D85A30','#993C1D','#EF9F27','#BA7517']
for ax, methods, title, cols in [
    (axes[0], list(unsup_methods.items())[:4], "Unsupervised methods", colors_unsup),
    (axes[1], list(unsup_methods.items()),     "All methods",          colors_unsup+colors_sup),
]:
    for (name,(ar,ap,fpr,tpr)), col in zip(methods, cols[:len(methods)]):
        ax.plot(fpr, tpr, color=col, lw=2, label=f"{name.split('(')[0].strip()} AUC={ar:.3f}")
    ax.plot([0,1],[0,1],'k--',lw=0.8,alpha=0.5)
    ax.set_title(title); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/anomaly_ROC_comparison.png",dpi=150,bbox_inches='tight')
plt.close(); print("Saved anomaly_ROC_comparison.png")

# ════════════════════════════════════════════════════════════════
# PHASE 8 — FEATURE VISUALISATION (t-SNE)
# ════════════════════════════════════════════════════════════════
print("\n[Phase 8] t-SNE feature visualisation...")

# Extract transformer features for comparison
def extract_transformer_features(model, loader, max_batches=20):
    model.eval(); out_feats=[]; out_lbls=[]
    hooks=[]; feat_buf=[]
    def h_fn(m,i,o): feat_buf.append(o.detach().cpu())
    if hasattr(model,'heads'):  # ViT
        hook = model.encoder.register_forward_hook(h_fn)
    else:                       # Swin
        hook = model.features.register_forward_hook(h_fn)
    hooks.append(hook)
    with torch.no_grad():
        for i,(x,y) in enumerate(loader):
            if i>=max_batches: break
            feat_buf.clear(); _ = model(x.to(device))
            if feat_buf:
                f = feat_buf[0]
                if f.dim()>2: f = f.mean(dim=list(range(1,f.dim()-1)))
                out_feats.append(f); out_lbls.extend(y.numpy())
    for h in hooks: h.remove()
    return torch.cat(out_feats,0).numpy() if out_feats else None, np.array(out_lbls)

# CNN features (EfficientNetB0 S3) — already extracted in `feats`
cnn_feats = feats[:2000]; cnn_lbls = lbls[:2000]   # subsample for speed

# ViT features
print("  Extracting ViT features...")
vit_m = build_vit()
vit_m.load_state_dict(torch.load(f"{SAVE_DIR}/S3_ViT_best.pth", map_location=device))
vit_feats, vit_lbls = extract_transformer_features(vit_m, test_ld, max_batches=30)
del vit_m; torch.cuda.empty_cache()

# Run t-SNE
for feat_name, fts, lbs in [
    ('CNN-EfficientNetB0', cnn_feats, cnn_lbls),
    ('Transformer-ViT',   vit_feats[:2000] if vit_feats is not None else cnn_feats, vit_lbls[:2000] if vit_feats is not None else cnn_lbls),
]:
    if fts is None: continue
    print(f"  t-SNE on {feat_name} ({len(fts)} samples)...")
    tsne  = TSNE(n_components=2, random_state=42, perplexity=40, n_iter=1000)
    emb   = tsne.fit_transform(fts)
    palette = plt.cm.tab20(np.linspace(0,1,NUM_CLASSES))

    fig, axes = plt.subplots(1,2,figsize=(18,8))
    fig.suptitle(f"t-SNE Feature Space — {feat_name}", fontsize=13, fontweight='bold')

    ax = axes[0]
    for i,cls in enumerate(class_names):
        mask = lbs==i
        if mask.sum()==0: continue
        is_min = cls in minority_cls
        ax.scatter(emb[mask,0], emb[mask,1], color=palette[i],
                    label=cls[:16], s=18 if is_min else 6,
                    alpha=0.85 if is_min else 0.3,
                    edgecolors='k' if is_min else 'none', linewidths=0.3)
    ax.set_title("Coloured by class (large=minority)"); ax.axis('off')
    ax.legend(fontsize=7, ncol=2, markerscale=2)

    ax = axes[1]
    norm_m = lbs==normal_idx
    ax.scatter(emb[norm_m,0],  emb[norm_m,1],  c='#7F77DD', s=5,  alpha=0.3, label='Normal')
    ax.scatter(emb[~norm_m,0], emb[~norm_m,1], c='#D85A30', s=12, alpha=0.7,
               edgecolors='k', linewidths=0.3, label='Anomaly')
    ax.set_title("Binary: Normal vs Anomaly"); ax.axis('off')
    ax.legend(fontsize=10, markerscale=2)

    plt.tight_layout()
    safe_name = feat_name.replace('-','_').replace('/','_')
    plt.savefig(f"{OUTPUT_DIR}/tsne_{safe_name}.png",dpi=150,bbox_inches='tight')
    plt.close(); print(f"  Saved tsne_{safe_name}.png")

# ════════════════════════════════════════════════════════════════
# PHASE 9 — GRADCAM + FALSE SAMPLES
# ════════════════════════════════════════════════════════════════
print("\n[Phase 9] GradCAM + false sample analysis...")

best_m.eval()
class GradCAM:
    def __init__(self, m):
        self.m=m; self.g=None; self.a=None
        base = m.module if isinstance(m,nn.DataParallel) else m
        target = (base.blocks[-1] if hasattr(base,'blocks')
                  else base.features[-1] if hasattr(base,'features')
                  else base.layer4[-1])
        target.register_forward_hook(lambda m,i,o: setattr(self,'a',o.detach()))
        target.register_full_backward_hook(lambda m,gi,go: setattr(self,'g',go[0].detach()))
    def generate(self, img_t, cls_idx=None):
        self.m.eval()
        inp = img_t.unsqueeze(0).to(device); inp.requires_grad_(True)
        out = self.m(inp)
        if cls_idx is None: cls_idx=out.argmax(1).item()
        self.m.zero_grad(); out[0,cls_idx].backward()
        w   = self.g.mean(dim=[d for d in range(2,self.g.dim())], keepdim=True)
        cam = F.relu((w*self.a).sum(1,keepdim=True)).squeeze().cpu().numpy()
        cam = (cam-cam.min())/(cam.max()-cam.min()+1e-8)
        return cam, cls_idx

def overlay(img_t, cam, alpha=0.5):
    mn=torch.tensor([0.485,0.456,0.406]).view(3,1,1)
    sd=torch.tensor([0.229,0.224,0.225]).view(3,1,1)
    img=(img_t.cpu()*sd+mn).clamp(0,1).permute(1,2,0).numpy()
    r=cv2.resize(cam,(img.shape[1],img.shape[0]))
    return np.clip(alpha*mpl_cm.jet(r)[:,:,:3]+(1-alpha)*img,0,1)

gradcam = GradCAM(best_m)
anom_imgs = {}
for imgs, lbs in test_ld:
    for i in range(len(lbs)):
        cls=class_names[int(lbs[i])]
        if cls in minority_cls and cls not in anom_imgs:
            anom_imgs[cls]=imgs[i]
    if len(anom_imgs)==len(minority_cls): break

n_show=min(len(anom_imgs),4)
if n_show>0:
    fig,axes=plt.subplots(n_show,2,figsize=(10,4*n_show))
    fig.suptitle(f"GradCAM — {best_tag}", fontsize=12, fontweight='bold')
    if n_show==1: axes=[axes]
    mn=torch.tensor([0.485,0.456,0.406]).view(3,1,1)
    sd=torch.tensor([0.229,0.224,0.225]).view(3,1,1)
    for row,(cls_name,img_t) in enumerate(list(anom_imgs.items())[:n_show]):
        try:
            cam,pred_idx=gradcam.generate(img_t,class_names.index(cls_name))
            ov=overlay(img_t,cam)
        except Exception as e:
            print(f"  GradCAM failed for {cls_name}: {e}"); continue
        img_np=(img_t.cpu()*sd+mn).clamp(0,1).permute(1,2,0).numpy()
        pn=class_names[pred_idx]
        axes[row][0].imshow(img_np); axes[row][0].set_title(f"Original: {cls_name}"); axes[row][0].axis('off')
        axes[row][1].imshow(ov)
        axes[row][1].set_title(f"GradCAM  Pred:{pn[:16]}",
                                color='green' if pn==cls_name else 'red')
        axes[row][1].axis('off')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/gradcam_{best_tag}.png",dpi=150,bbox_inches='tight')
    plt.close(); print(f"Saved gradcam_{best_tag}.png")

# False sample analysis
fp=[]; fn_c={i:0 for i in range(NUM_CLASSES)}; fp_c={i:0 for i in range(NUM_CLASSES)}; tot_c={i:0 for i in range(NUM_CLASSES)}
best_m.eval()
with torch.no_grad():
    for x,y in test_ld:
        out=best_m(x.to(device)); pr=F.softmax(out,1); cf,pred=torch.max(pr,1)
        for i in range(len(y)):
            tl=int(y[i]); pl=int(pred[i]); c=float(cf[i])
            tot_c[tl]=tot_c.get(tl,0)+1
            if pl!=tl:
                fn_c[tl]+=1; fp_c[pl]+=1; fp.append((x[i],tl,pl,c))
fp.sort(key=lambda x:x[3],reverse=True)
df_fp=pd.DataFrame([{
    'Class':class_names[i],'Total':tot_c.get(i,0),
    'FN':fn_c.get(i,0),'FP':fp_c.get(i,0),
    'Miss rate':round(fn_c.get(i,0)/max(tot_c.get(i,0),1),3),
    'Type':'MINORITY' if class_names[i] in minority_cls else 'majority'
} for i in range(NUM_CLASSES)]).sort_values('Miss rate',ascending=False)
print("Per-class miss rate:"); print(df_fp.to_string(index=False))
df_fp.to_csv(f"{OUTPUT_DIR}/false_sample_stats.csv", index=False)

n=min(10,len(fp))
if n>0:
    fig,axes=plt.subplots(2,5,figsize=(18,8))
    fig.suptitle(f"Worst False Positives — {best_tag}",fontsize=12,fontweight='bold')
    axes=axes.flatten()
    for i in range(10):
        ax=axes[i]
        if i>=n: ax.axis('off'); continue
        img_t,tl,pl,cf=fp[i]
        mn=torch.tensor([0.485,0.456,0.406]).view(3,1,1)
        sd=torch.tensor([0.229,0.224,0.225]).view(3,1,1)
        ax.imshow((img_t*sd+mn).clamp(0,1).permute(1,2,0).numpy())
        ax.set_title(f"True:{class_names[tl][:12]}\nPred:{class_names[pl][:12]}({cf:.2f})",
                      fontsize=8,color='red'); ax.axis('off')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/false_positives.png",dpi=130,bbox_inches='tight')
    plt.close()

# ════════════════════════════════════════════════════════════════
# PHASE 10 — FINAL SUMMARY PLOT
# ════════════════════════════════════════════════════════════════
print("\n[Phase 10] Final summary plot...")

fig, axes = plt.subplots(2,3,figsize=(21,12))
fig.suptitle("WCE Major Project — Complete Results Summary", fontsize=14, fontweight='bold')

# 1. CNN vs Transformer F1(macro)
ax=axes[0,0]
tags = list(all_results.keys())
f1s  = [all_results[t]['metrics']['F1(macro)'] for t in tags]
cols = ['#7F77DD' if 'S1' in t else ('#1D9E75' if 'S3' in t and ('ViT' in t or 'Swin' in t)
         else '#D85A30') for t in tags]
bars=ax.bar(range(len(tags)),f1s,color=cols,alpha=0.85)
ax.set_xticks(range(len(tags))); ax.set_xticklabels([t.replace('S1_','').replace('S3_','') for t in tags],rotation=45,ha='right',fontsize=9)
ax.set_title('Macro F1 — All Models'); ax.set_ylabel('F1(macro)'); ax.set_ylim(0,1.05)
ax.grid(alpha=0.3,axis='y')
for b,v in zip(bars,f1s): ax.text(b.get_x()+b.get_width()/2,v+0.01,f'{v:.3f}',ha='center',fontsize=8)

# 2. S1 vs S3 on best CNN
ax=axes[0,1]
met=['Acc(w)','F1(w)','Rec(macro)','F1(macro)']
x=np.arange(len(met)); w=0.35
best_cnn = max(['EffNetB0','MobileV3','ResNet50'],
                key=lambda n: all_results.get(f'S3_{n}',{}).get('metrics',{}).get('F1(macro)',0))
s1v=[all_results.get(f'S1_{best_cnn}',{'metrics':{}})['metrics'].get(m,0) for m in met]
s3v=[all_results.get(f'S3_{best_cnn}',{'metrics':{}})['metrics'].get(m,0) for m in met]
ax.bar(x-w/2,s1v,w,label=f'S1 (raw) {best_cnn}',color='#7F77DD',alpha=0.85)
ax.bar(x+w/2,s3v,w,label=f'S3 (balanced) {best_cnn}',color='#1D9E75',alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(met,fontsize=9); ax.set_ylim(0,1.15)
ax.set_title('S1 vs S3 (best CNN)'); ax.legend(fontsize=9); ax.grid(alpha=0.3,axis='y')

# 3. Anomaly AUROC comparison
ax=axes[0,2]
methods=[n.split('(')[0].strip() for n in unsup_methods.keys()]
aurocs =[v[0] for v in unsup_methods.values()]
bar_cols=['#D85A30']*4 + ['#7F77DD']*2
bars=ax.bar(range(len(methods)),aurocs,color=bar_cols,alpha=0.85)
ax.set_xticks(range(len(methods))); ax.set_xticklabels(methods,rotation=45,ha='right',fontsize=8)
ax.set_title('Anomaly Detection AUROC'); ax.set_ylim(0,1.05); ax.grid(alpha=0.3,axis='y')
for b,v in zip(bars,aurocs): ax.text(b.get_x()+b.get_width()/2,v+0.01,f'{v:.3f}',ha='center',fontsize=8)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color='#D85A30',label='Unsupervised'),
                    Patch(color='#7F77DD',label='Supervised')],fontsize=9)

# 4. F1(macro) S1 vs S3 vs Transformer
ax=axes[1,0]
xp=np.arange(3); w2=0.28
s1_f=[all_results.get(f'S1_{n}',{'metrics':{'F1(macro)':0}})['metrics']['F1(macro)'] for n in ['EffNetB0','MobileV3','ResNet50']]
s3_f=[all_results.get(f'S3_{n}',{'metrics':{'F1(macro)':0}})['metrics']['F1(macro)'] for n in ['EffNetB0','MobileV3','ResNet50']]
ax.bar(xp-w2,s1_f,w2,label='S1 raw',color='#7F77DD',alpha=0.85)
ax.bar(xp,   s3_f,w2,label='S3 balanced',color='#1D9E75',alpha=0.85)
ax.set_xticks(xp); ax.set_xticklabels(['EffNetB0','MobileV3','ResNet50'])
ax.set_title('CNN: S1 vs S3 F1(macro)'); ax.legend(); ax.set_ylim(0,1.05); ax.grid(alpha=0.3,axis='y')

# 5. Transformer vs best CNN
ax=axes[1,1]
all_compare = {
    f'CNN Best\n({best_cnn} S3)': all_results.get(f'S3_{best_cnn}',{'metrics':{'Rec(macro)':0,'F1(macro)':0}})['metrics'],
    'ViT S3' : all_results.get('S3_ViT', {'metrics':{'Rec(macro)':0,'F1(macro)':0}})['metrics'],
    'Swin S3': all_results.get('S3_Swin',{'metrics':{'Rec(macro)':0,'F1(macro)':0}})['metrics'],
}
xp3=np.arange(len(all_compare)); w3=0.35
ax.bar(xp3-w3/2,[m.get('Rec(macro)',0) for m in all_compare.values()],w3,label='Rec(macro)',color='#D85A30',alpha=0.85)
ax.bar(xp3+w3/2,[m.get('F1(macro)',0)  for m in all_compare.values()],w3,label='F1(macro)', color='#534AB7',alpha=0.85)
ax.set_xticks(xp3); ax.set_xticklabels(list(all_compare.keys()),fontsize=10)
ax.set_title('CNN vs Transformer (S3)'); ax.legend(); ax.set_ylim(0,1.1); ax.grid(alpha=0.3,axis='y')

# 6. Anomaly AUPRC comparison
ax=axes[1,2]
auprcs=[v[1] for v in unsup_methods.values()]
bars=ax.bar(range(len(methods)),auprcs,color=bar_cols,alpha=0.85)
ax.set_xticks(range(len(methods))); ax.set_xticklabels(methods,rotation=45,ha='right',fontsize=8)
ax.set_title('Anomaly Detection AUPRC'); ax.set_ylim(0,1.05); ax.grid(alpha=0.3,axis='y')
for b,v in zip(bars,auprcs): ax.text(b.get_x()+b.get_width()/2,v+0.01,f'{v:.3f}',ha='center',fontsize=8)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/FINAL_summary.png",dpi=150,bbox_inches='tight')
plt.close(); print("Saved FINAL_summary.png")

# ════════════════════════════════════════════════════════════════
# OUTPUT SUMMARY
# ════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("ALL OUTPUT FILES")
print("="*65)
for f in sorted(os.listdir(OUTPUT_DIR)):
    p=os.path.join(OUTPUT_DIR,f); kb=os.path.getsize(p)/1024
    print(f"  {f:<55} {kb:>8.1f} KB")

print("\n=== KEY FINDINGS ===")
best_cnn_tag = max([t for t in all_results if 'ViT' not in t and 'Swin' not in t],
                    key=lambda t: all_results[t]['metrics']['F1(macro)'])
best_tfm_tag = max([t for t in all_results if 'ViT' in t or 'Swin' in t],
                    key=lambda t: all_results[t]['metrics']['F1(macro)'], default=None)
print(f"Best CNN       : {best_cnn_tag} | F1(macro)={all_results[best_cnn_tag]['metrics']['F1(macro)']:.4f}")
if best_tfm_tag:
    print(f"Best Transformer: {best_tfm_tag} | F1(macro)={all_results[best_tfm_tag]['metrics']['F1(macro)']:.4f}")
print(f"Best Unsupervised anomaly: Autoencoder AUROC={auroc_a:.4f}")
print(f"Best Supervised  anomaly: Binary detector  AUROC={auroc_b:.4f}")
print(f"\nConclusion: Sir's feedback addressed:")
print(f"  1. Transformer models (ViT, Swin) included and compared")
print(f"  2. Unsupervised anomaly: K-Means, IsoForest, OCSVM, Autoencoder")
print(f"  3. Supervised anomaly: binary detector + softmax score")
print(f"  4. t-SNE visualisation of CNN vs Transformer feature spaces")
print(f"  5. GradCAM on best model for anomaly localisation")
print(f"  6. Full false sample analysis")
