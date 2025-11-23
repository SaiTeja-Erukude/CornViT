import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import sys
import os
from timm.loss import SoftTargetCrossEntropy
from timm.scheduler import CosineLRScheduler
from timm.utils import accuracy
import matplotlib.pyplot as plt
import json
from datetime import datetime


# ============================================================
# SETUP: Clone and import from Microsoft CvT repository
# ============================================================
"""
First, clone the Microsoft CvT repository:
    git clone https://github.com/microsoft/CvT.git
    cd CvT
    pip install -r requirements.txt
"""

BASE_DIR = "path_to_CornViT"

# Add the CvT repo to Python path
CVT_REPO_PATH = f"{BASE_DIR}/CvT"

if not os.path.exists(CVT_REPO_PATH):
    print(f"❌ CvT repository not found at {CVT_REPO_PATH}")
    print("Please clone it: git clone https://github.com/microsoft/CvT.git")
    sys.exit(1)

# Fix torch._six compatibility BEFORE importing
print("Applying compatibility fixes for newer PyTorch versions...")
cls_cvt_path = os.path.join(CVT_REPO_PATH, "lib", "models", "cls_cvt.py")

if os.path.exists(cls_cvt_path):
    with open(cls_cvt_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix 1: Replace torch._six import
    if "from torch._six import container_abcs" in content:
        content = content.replace(
            "from torch._six import container_abcs",
            "import collections.abc as container_abcs"
        )
        
        # Fix 2: Replace 'is' with '==' for string comparison
        content = content.replace(
            "or pretrained_layers[0] is '*'",
            "or pretrained_layers[0] == '*'"
        )
        
        with open(cls_cvt_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print("✅ Applied compatibility patches to cls_cvt.py")
    else:
        print("✅ Compatibility patches already applied")
else:
    print(f"❌ Could not find cls_cvt.py at {cls_cvt_path}")
    sys.exit(1)

# Now import
sys.path.insert(0, CVT_REPO_PATH)

# Suppress the SyntaxWarning
import warnings
warnings.filterwarnings('ignore', category=SyntaxWarning)

from lib.models import cls_cvt
from lib.config import config, update_config
print("✅ Successfully imported Microsoft CvT models")


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR        = f"{BASE_DIR}/stage1/data"
BATCH_SIZE      = 32
IMG_SIZE        = 384
NUM_CLASSES     = 2
NUM_EPOCHS      = 100
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
PRETRAINED_PATH = f"{BASE_DIR}/CvT-13-384x384-IN-22k.pth"

# Create output directory for saving results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = f"metrics/cvt13_run_{timestamp}"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Metrics will be saved to: {OUTPUT_DIR}")


# ============================================================
# DATASET & AUGMENTATION
# ============================================================

train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(f"{DATA_DIR}/train", transform=train_transforms)
val_dataset = datasets.ImageFolder(f"{DATA_DIR}/val", transform=val_transforms)

train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    num_workers=0, 
    pin_memory=True,
    drop_last=True
)
val_loader = DataLoader(
    val_dataset, 
    batch_size=BATCH_SIZE,
    shuffle=False, 
    num_workers=0, 
    pin_memory=True,
    drop_last=True
)


# ============================================================
# MODEL SETUP - Using Microsoft CvT Implementation
# ============================================================

# Load the CvT-13 config from the repository
cvt_config_path = os.path.join(CVT_REPO_PATH, "experiments", "imagenet", "cvt", "cvt-13-384x384.yaml")

if not os.path.exists(cvt_config_path):
    print(f"⚠️ Config file not found at {cvt_config_path}")
    print("Available configs:")
    config_dir = os.path.join(CVT_REPO_PATH, "experiments", "imagenet", "cvt")
    if os.path.exists(config_dir):
        for f in os.listdir(config_dir):
            if f.endswith('.yaml'):
                print(f"  - {f}")
    sys.exit(1)

print(f"Loading config from: {cvt_config_path}")

# Load config directly using merge_from_file
config.defrost()
config.merge_from_file(cvt_config_path)

# Update the number of classes for our task
config.MODEL.NUM_CLASSES = NUM_CLASSES
config.MODEL.PRETRAINED = ''  # We'll load weights manually
config.freeze()

print("Creating CvT-13 model...")
# Create model using the official CvT architecture
model = cls_cvt.get_cls_model(config)
model = model.to(DEVICE)

# Load pretrained weights
if os.path.exists(PRETRAINED_PATH):
    print(f"Loading pretrained weights from {PRETRAINED_PATH}")
    try:
        checkpoint = torch.load(PRETRAINED_PATH, map_location=DEVICE)
        
        # Handle different checkpoint formats
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Remove 'module.' prefix if present
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace("module.", "")
            new_state_dict[name] = v
        
        # Remove head layers from pretrained weights (they have different dimensions)
        filtered_state_dict = {k: v for k, v in new_state_dict.items() if 'head' not in k}
        
        # Load weights - strict=False will only load matching layers
        missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
        
        # Count how many weights were actually loaded
        loaded_keys = [k for k in filtered_state_dict.keys() if k in model.state_dict()]
        print(f"✅ Loaded pretrained weights: {len(loaded_keys)} layers from backbone")
        print(f"   Head layer initialized randomly for {NUM_CLASSES} classes")
        
        # Show what's missing (should only be head-related)
        head_missing = [k for k in missing_keys if 'head' in k]
        other_missing = [k for k in missing_keys if 'head' not in k]
        
        if other_missing:
            print(f"⚠️  Warning - Missing non-head keys: {other_missing}")
        if unexpected_keys:
            print(f"⚠️  Unexpected keys: {unexpected_keys}")
            
    except Exception as e:
        print(f"⚠️ Error loading pretrained weights: {e}")
        import traceback
        traceback.print_exc()
        print("Continuing with random initialization...")
else:
    print(f"⚠️ Pretrained weights not found at {PRETRAINED_PATH}")
    print("Training from scratch...")

# Freeze backbone - only train the head for faster training and less overfitting
print("Freezing backbone layers (keeping only head trainable)...")
for name, param in model.named_parameters():
    if "head" not in name:
        param.requires_grad = False

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Trainable parameters: {trainable_params:,} / {total_params:,}")
print(f"Frozen parameters: {total_params - trainable_params:,}")


# ============================================================
# OPTIMIZER AND LOSS
# ============================================================

optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()), 
    lr=1e-4, 
    weight_decay=0.05
)

criterion = SoftTargetCrossEntropy()

lr_scheduler = CosineLRScheduler(
    optimizer,
    t_initial=NUM_EPOCHS,
    lr_min=1e-6,
    warmup_t=5,
    warmup_lr_init=1e-5,
)


# ============================================================
# TRAINING & VALIDATION LOOP
# ============================================================

def train_one_epoch(epoch, history):
    model.train()
    total_loss, total_acc = 0, 0
    
    for images, targets in train_loader:
        images, targets = images.to(DEVICE), targets.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        acc1, _ = accuracy(outputs, targets.argmax(dim=1), topk=(1, 5))
        total_loss += loss.item()
        total_acc += acc1.item()

    avg_loss = total_loss / len(train_loader)
    avg_acc = total_acc / len(train_loader)
    
    history['train_loss'].append(avg_loss)
    history['train_acc'].append(avg_acc)
    history['learning_rate'].append(optimizer.param_groups[0]['lr'])
    
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | Train Loss: {avg_loss:.4f} | Train Acc: {avg_acc:.2f}% | LR: {optimizer.param_groups[0]['lr']:.6f}")
    return avg_loss, avg_acc


def validate(epoch, history):
    model.eval()
    total_loss, total_acc = 0, 0
    
    with torch.no_grad():
        for images, targets in val_loader:
            images, targets = images.to(DEVICE), targets.to(DEVICE)
            outputs = model(images)
            loss = nn.CrossEntropyLoss()(outputs, targets)
            acc1, _ = accuracy(outputs, targets, topk=(1, 5))
            
            total_loss += loss.item()
            total_acc += acc1.item()
    
    avg_loss = total_loss / len(val_loader)
    avg_acc = total_acc / len(val_loader)
    
    history['val_loss'].append(avg_loss)
    history['val_acc'].append(avg_acc)
    
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | Val Loss: {avg_loss:.4f} | Val Acc: {avg_acc:.2f}%")
    return avg_acc


def plot_training_history(history, save_path):
    """Plot and save training metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot 1: Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Accuracy
    axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0, 1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Learning Rate
    axes[1, 0].plot(epochs, history['learning_rate'], 'g-', linewidth=2)
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Learning Rate', fontsize=12)
    axes[1, 0].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Val Acc vs Train Acc (Overfitting check)
    axes[1, 1].plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    axes[1, 1].plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    gap = [t - v for t, v in zip(history['train_acc'], history['val_acc'])]
    axes[1, 1].fill_between(epochs, history['val_acc'], history['train_acc'], 
                            alpha=0.3, color='orange', label='Overfitting Gap')
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1, 1].set_title('Overfitting Analysis', fontsize=14, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Training plots saved to: {save_path}")
    plt.close()


def save_training_summary(history, best_acc, save_path):
    """Save training summary as JSON"""
    summary = {
        'config': {
            'model': 'CvT-13',
            'batch_size': BATCH_SIZE,
            'img_size': IMG_SIZE,
            'num_classes': NUM_CLASSES,
            'num_epochs': NUM_EPOCHS,
            'device': DEVICE,
            'pretrained': PRETRAINED_PATH,
        },
        'final_metrics': {
            'best_val_accuracy': best_acc,
            'final_train_loss': history['train_loss'][-1],
            'final_train_acc': history['train_acc'][-1],
            'final_val_loss': history['val_loss'][-1],
            'final_val_acc': history['val_acc'][-1],
        },
        'history': history
    }
    
    with open(save_path, 'w') as f:
        json.dump(summary, f, indent=4)
    
    print(f"💾 Training summary saved to: {save_path}")


# ============================================================
# MAIN TRAINING LOOP
# ============================================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60 + "\n")

    # Initialize history tracking
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rate': []
    }
    
    best_acc = 0.0
    best_epoch = 0

    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_one_epoch(epoch, history)
        val_acc = validate(epoch, history)
        lr_scheduler.step(epoch + 1)

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch + 1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'history': history,
            }, os.path.join(OUTPUT_DIR, "best_model.pth"))
            print(f"✅ Saved best model at epoch {epoch+1} with val acc {best_acc:.2f}%\n")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'history': history,
            }, os.path.join(OUTPUT_DIR, f"checkpoint_epoch_{epoch+1}.pth"))
            print(f"💾 Checkpoint saved at epoch {epoch+1}\n")
        
        # Plot and save metrics every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == NUM_EPOCHS - 1:
            plot_training_history(history, os.path.join(OUTPUT_DIR, "training_metrics.png"))

    # Final summary
    print("="*60)
    print(f"🎉 Training complete!")
    print(f"Best validation accuracy: {best_acc:.2f}% at epoch {best_epoch}")
    print(f"Final train accuracy: {history['train_acc'][-1]:.2f}%")
    print(f"Final val accuracy: {history['val_acc'][-1]:.2f}%")
    print("="*60)
    
    # Save final training summary
    save_training_summary(history, best_acc, os.path.join(OUTPUT_DIR, "training_summary.json"))
    
    # Save final plot
    plot_training_history(history, os.path.join(OUTPUT_DIR, "final_training_metrics.png"))
    
    print(f"\n📁 All outputs saved to: {OUTPUT_DIR}")