# code3_final.py
import os
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

import matplotlib; matplotlib.use("Agg")
import random
import copy
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, f1_score, confusion_matrix, roc_curve, auc
from sklearn.cluster import KMeans
import torch.nn.functional as F
from PIL import Image

warnings.filterwarnings("ignore")
device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
print(f"Device: {device}")

# ======================== GLOBALS (HPC Paths) ========================
DATA_DIR_D1 = '/home/rahuldixit/aksh/dl_project/dataset/extracted_images'
OUTPUT_DIR  = '/home/rahuldixit/aksh/dl_project/outputs_code3'
SAVE_DIR    = '/home/rahuldixit/aksh/dl_project/saved_models_code3'

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)

# ----------------- PARAMS -----------------
IMG_SIZE         = 224
BATCH_SIZE       = 32
EPOCHS           = 15 # Optimized for stable convergence
UNDER_SAMPLE_CAP = 300
AUG_TARGET       = 600
FOCAL_GAMMA      = 2.0
print(f"IMG={IMG_SIZE}  BATCH={BATCH_SIZE}  EPOCHS={EPOCHS}  FOCAL_GAMMA={FOCAL_GAMMA}")

# ======================== DATA LOADING & S3 BALANCING ========================
tf_raw = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

raw_ds = ImageFolder(root=DATA_DIR_D1, transform=tf_raw)
classes = raw_ds.classes

class_counts = {c: 0 for c in classes}
class_indices = {c: [] for c in classes}
for idx, (_, label) in enumerate(raw_ds.samples):
    c = classes[label]
    class_counts[c] += 1
    class_indices[c].append(idx)

# Random Split (Train 70%, Val 15%, Test 15%)
random.seed(42)
train_idx, val_idx, test_idx = [], [], []

for c in classes:
    idxs = class_indices[c]
    random.shuffle(idxs)
    n = len(idxs)
    n_tr = int(n * 0.70)
    n_vl = int(n * 0.15)
    train_idx.extend(idxs[:n_tr])
    val_idx.extend(idxs[n_tr:n_tr+n_vl])
    test_idx.extend(idxs[n_tr+n_vl:])

class IndexDS(Dataset):
    def __init__(self, full_ds, indices):
        self.ds = full_ds
        self.idxs = indices
    def __len__(self): return len(self.idxs)
    def __getitem__(self, i):
        path, lbl = self.ds.samples[self.idxs[i]]
        img = Image.open(path).convert('RGB')
        img = self.ds.transform(img)
        return img, lbl, self.idxs[i]

shared_test = IndexDS(raw_ds, test_idx)
shared_val  = IndexDS(raw_ds, val_idx)

# S3 Balancing Logic
tf_aug = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.5),
    T.RandomRotation(15),
    T.ColorJitter(brightness=0.1, contrast=0.1),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

s3_train_ds_list = []
for c in classes:
    c_idx = classes.index(c)
    pool = [i for i in train_idx if raw_ds.targets[i] == c_idx]
    
    if len(pool) > UNDER_SAMPLE_CAP:
        selected = random.sample(pool, UNDER_SAMPLE_CAP)
        for i in selected: s3_train_ds_list.append((raw_ds.samples[i][0], c_idx, False))
    else:
        for i in pool: s3_train_ds_list.append((raw_ds.samples[i][0], c_idx, False))
        needed = AUG_TARGET - len(pool)
        if needed > 0:
            for _ in range(needed):
                s3_train_ds_list.append((raw_ds.samples[random.choice(pool)][0], c_idx, True))

class AugListDS(Dataset):
    def __init__(self, data_list, tf_raw, tf_aug):
        self.data = data_list
        self.tf_raw = tf_raw
        self.tf_aug = tf_aug
    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        path, lbl, is_aug = self.data[i]
        img = Image.open(path).convert('RGB')
        img = self.tf_aug(img) if is_aug else self.tf_raw(img)
        return img, lbl, i

train_s3 = AugListDS(s3_train_ds_list, tf_raw, tf_aug)

kw = dict(num_workers=8, pin_memory=True)
tr_ld = DataLoader(train_s3, BATCH_SIZE, shuffle=True, **kw)
vl_ld = DataLoader(shared_val, BATCH_SIZE, shuffle=False, **kw)
ts_ld = DataLoader(shared_test, BATCH_SIZE, shuffle=False, **kw)

# ======================== FOCAL LOSS (WEIGHTS REMOVED) ========================
# FIXED: We removed manual alpha weights because physical S3 balancing handles it.
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma
    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        return (((1 - pt) ** self.gamma) * BCE_loss).mean()

criterion = FocalLoss(gamma=FOCAL_GAMMA)

# ======================== ARCHITECTURES ========================
def build_resnet50():
    m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    m.fc = nn.Linear(m.fc.in_features, len(classes))
    return m.to(device)

def build_vit():
    m = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
    m.heads.head = nn.Linear(m.heads.head.in_features, len(classes))
    return m.to(device)

def build_swin():
    m = models.swin_t(weights=models.Swin_T_Weights.DEFAULT)
    m.head = nn.Linear(m.head.in_features, len(classes))
    return m.to(device)

builders = {'ResNet50': build_resnet50, 'ViT': build_vit, 'Swin': build_swin}

# ======================== TRAINING LOOP WITH PLOTTING ========================
def train_model(model, name):
    opt = optim.AdamW(model.parameters(), lr=1e-4)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=2)
    history = {'loss': [], 'val_f1': [], 'val_acc': []}
    best_f1 = 0.0
    
    print(f"\n--- Training {name} ---")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for x, y, _ in tr_ld:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            
        model.eval()
        preds, t_true = [], []
        with torch.no_grad():
            for x, y, _ in vl_ld:
                out = model(x.to(device))
                preds.extend(out.argmax(1).cpu().numpy())
                t_true.extend(y.numpy())
        
        vacc = np.mean(np.array(preds) == np.array(t_true))
        vf1 = f1_score(t_true, preds, average='macro')
        history['loss'].append(total_loss/len(tr_ld))
        history['val_f1'].append(vf1)
        history['val_acc'].append(vacc)
        
        print(f" Ep {epoch+1}/{EPOCHS} | Loss: {history['loss'][-1]:.4f} | Val F1: {vf1:.4f} | Val Acc: {vacc:.4f}")
        
        sched.step(vf1)
        if vf1 > best_f1:
            best_f1 = vf1
            torch.save(model.state_dict(), f"{SAVE_DIR}/{name}_best.pth")

    # Save Plots for Report
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Loss')
    plt.title(f'{name} Training Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history['val_acc'], label='Accuracy')
    plt.plot(history['val_f1'], label='Macro F1')
    plt.title(f'{name} Validation Metrics')
    plt.legend()
    plt.savefig(f"{OUTPUT_DIR}/{name}_training_history.png")
    plt.close()
    
    return model

# ======================== EXECUTION ========================
results = {}
trained_models = {}

for mname, builder in builders.items():
    m = train_model(builder(), mname)
    trained_models[mname] = m
    
    # Eval & Error Tracking
    m.eval()
    p_all, t_all, idx_all = [], [], []
    with torch.no_grad():
        for x, y, idxs in ts_ld:
            out = m(x.to(device))
            p_all.extend(out.argmax(1).cpu().numpy())
            t_all.extend(y.numpy())
            idx_all.extend(idxs.numpy())
            
    # Save False Samples for GradCAM analysis
    with open(f"{OUTPUT_DIR}/{mname}_false_samples.txt", "w") as f:
        f.write("Path | True | Pred\n")
        for i in range(len(t_all)):
            if t_all[i] != p_all[i]:
                f.write(f"{raw_ds.samples[idx_all[i]][0]} | {classes[t_all[i]]} | {classes[p_all[i]]}\n")
    
    results[mname] = classification_report(t_all, p_all, target_names=classes, output_dict=True)
    print(f"\n{mname} Final Report:\n", classification_report(t_all, p_all, target_names=classes))

# ======================== UNSUPERVISED ANOMALY (ResNet50) ========================
print("\n--- Running Unsupervised K-Means Analysis ---")
best_m = trained_models['ResNet50']
best_m.eval()
best_m.fc = nn.Identity()

feats, bin_y = [], []
with torch.no_grad():
    for x, y, _ in ts_ld:
        f = best_m(x.to(device))
        feats.append(f.cpu().numpy())
        bin_y.extend([0 if yi == classes.index('Normal clean mucosa') else 1 for yi in y])

feats = np.concatenate(feats)
bin_y = np.array(bin_y)

# Find Normal Centroid
normal_feats = feats[bin_y == 0]
km = KMeans(n_clusters=1, n_init=10).fit(normal_feats)
dist = np.linalg.norm(feats - km.cluster_centers_, axis=1)

fpr, tpr, _ = roc_curve(bin_y, dist)
auc_score = auc(fpr, tpr)
print(f"Unsupervised Centroid AUROC: {auc_score:.4f}")

plt.figure()
plt.plot(fpr, tpr, label=f'AUC = {auc_score:.2f}')
plt.plot([0,1],[0,1], '--')
plt.title('Unsupervised Anomaly ROC')
plt.legend()
plt.savefig(f"{OUTPUT_DIR}/Unsupervised_Metric_ROC.png")
plt.close()

print("Pipeline Complete. Check outputs_code3 for graphs and error logs.")

