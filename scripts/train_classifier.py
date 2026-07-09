"""
Train a lightweight CNN binary classifier: triangle vs not-triangle.

Uses PyTorch with a simple CNN architecture that can classify 64x64 crops.
This replaces the template matching confidence with a learned classifier
that should dramatically improve precision while maintaining recall.

The model is small enough to run inference on all DB candidates quickly.
"""

import sys
import random
import numpy as np
import cv2
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class TriangleCNN(nn.Module):
    """
    Small CNN for triangle classification.
    Input: 64x64 grayscale (red-suppressed) image
    Output: probability of triangle presence
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # 64x64 -> 32x32
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # 32x32 -> 16x16
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # 16x16 -> 8x8
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # 8x8 -> 4x4
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class TriangleDataset(Dataset):
    """Load pre-generated crops for training."""

    def __init__(self, data_dir, augment=False, include_partials=None,
                 neg_ratio=3.0):
        self.augment = augment
        self.samples = []
        # Seeded RNG: dataset composition must be deterministic so that
        # validation metrics are comparable across training runs
        rng = random.Random(0)

        pos_dir = Path(data_dir) / 'positive'
        neg_dir = Path(data_dir) / 'negative'
        # Partial crops (triangles, just off-center) live OUTSIDE the
        # train/val split. Only include them in the training set — adding
        # them to both sets means validating on training images, which
        # inflates the reported F1/precision/recall.
        if include_partials is None:
            include_partials = augment
        partial_dir = Path(data_dir).parent / 'partial'

        for f in sorted(pos_dir.glob('*.png')):
            self.samples.append((str(f), 1))
        if include_partials and partial_dir.exists():
            for f in sorted(partial_dir.glob('*.png')):
                self.samples.append((str(f), 1))

        pos_count = len(self.samples)

        # Cap negatives at neg_ratio x positives, prioritizing hard negatives.
        # The WeightedRandomSampler balances batches regardless of pool size,
        # so a larger pool = more negative VARIETY per epoch = better
        # precision. neg_ratio<=0 means use every available negative.
        neg_files = sorted(neg_dir.glob('*.png'))
        # Identify hard negatives (from hard_negatives dir, symlinked into negative)
        hn_files = [f for f in neg_files if 'hn' in f.name]
        other_neg = [f for f in neg_files if 'hn' not in f.name]
        max_neg = int(pos_count * neg_ratio) if neg_ratio > 0 else len(neg_files)
        # Take all hard negatives first, fill rest with random regular negatives
        rng.shuffle(other_neg)
        selected_neg = hn_files + other_neg[:max(0, max_neg - len(hn_files))]
        for f in selected_neg:
            self.samples.append((str(f), 0))

        rng.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        # Load as BGR, apply red suppression (same as detection pipeline)
        bgr = cv2.imread(path)
        if bgr is None:
            # Return zeros if file can't be read
            return torch.zeros(1, 64, 64), label

        b = bgr[:, :, 0].astype(np.int16)
        g = bgr[:, :, 1].astype(np.int16)
        gray = np.minimum(b, g).astype(np.uint8)

        # Resize to 64x64 if needed
        if gray.shape[0] != 64 or gray.shape[1] != 64:
            gray = cv2.resize(gray, (64, 64))

        # Augmentation
        if self.augment:
            # Random brightness
            if random.random() < 0.3:
                delta = random.randint(-20, 20)
                gray = np.clip(gray.astype(np.int16) + delta, 0, 255).astype(np.uint8)

            # Random noise
            if random.random() < 0.2:
                noise = np.random.normal(0, 5, gray.shape).astype(np.int16)
                gray = np.clip(gray.astype(np.int16) + noise, 0, 255).astype(np.uint8)

            # Random small rotation (triangles can be slightly rotated)
            if random.random() < 0.3:
                angle = random.uniform(-5, 5)
                M = cv2.getRotationMatrix2D((32, 32), angle, 1.0)
                gray = cv2.warpAffine(gray, M, (64, 64),
                                       borderMode=cv2.BORDER_REFLECT)

            # Random translation (simulate off-center crops)
            if random.random() < 0.3:
                tx = random.randint(-6, 6)
                ty = random.randint(-6, 6)
                M = np.float32([[1, 0, tx], [0, 1, ty]])
                gray = cv2.warpAffine(gray, M, (64, 64),
                                       borderMode=cv2.BORDER_REFLECT)

            # Random horizontal flip
            if random.random() < 0.3:
                gray = cv2.flip(gray, 1)

        # Normalize to [0, 1]
        tensor = torch.from_numpy(gray.astype(np.float32) / 255.0).unsqueeze(0)
        return tensor, label


def train_model(data_dir, output_path, epochs=34, batch_size=32, lr=0.001,
                neg_ratio=0.0):
    """Train the triangle classifier.

    neg_ratio controls how many negatives enter the training pool
    (xPositives); higher = more negative variety = better precision.
    neg_ratio<=0 (the default) uses every available negative — a frozen-test
    sweep showed this lifts F1 0.449 -> 0.481 (precision 0.30 -> 0.33) over
    the old 3x cap, with recall preserved. The batch sampler still balances
    classes, so the larger pool only adds negative variety per epoch.
    """
    if not HAS_TORCH:
        print("PyTorch not available. Install with: pip install torch torchvision")
        return None

    device = torch.device('mps' if torch.backends.mps.is_available()
                          else 'cuda' if torch.cuda.is_available()
                          else 'cpu')
    print(f"Using device: {device}")

    # Load datasets
    train_dir = Path(data_dir) / 'train'
    val_dir = Path(data_dir) / 'val'

    # Train pool uses neg_ratio; val stays at the default 3x so its
    # composition doesn't shift with the experiment
    train_dataset = TriangleDataset(train_dir, augment=True, neg_ratio=neg_ratio)
    val_dataset = TriangleDataset(val_dir, augment=False)

    # Handle class imbalance: oversample positives
    pos_count = sum(1 for _, l in train_dataset.samples if l == 1)
    neg_count = len(train_dataset.samples) - pos_count
    print(f"Training: {pos_count} positive, {neg_count} negative")
    print(f"Validation: {len(val_dataset)} samples")

    # Weighted sampler for balanced batches
    weights = []
    for _, label in train_dataset.samples:
        weights.append(1.0 / pos_count if label == 1 else 1.0 / neg_count)
    sampler = torch.utils.data.WeightedRandomSampler(
        weights, len(weights), replacement=True
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                               sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=0)

    # Model, loss, optimizer
    model = TriangleCNN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    best_val_f1 = 0
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.float().to(device)

            optimizer.zero_grad()
            outputs = model(images).squeeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        scheduler.step()

        # Validation
        model.eval()
        val_tp, val_fp, val_fn, val_tn = 0, 0, 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.float().to(device)

                outputs = model(images).squeeze(1)
                preds = (torch.sigmoid(outputs) > 0.5).float()

                val_tp += ((preds == 1) & (labels == 1)).sum().item()
                val_fp += ((preds == 1) & (labels == 0)).sum().item()
                val_fn += ((preds == 0) & (labels == 1)).sum().item()
                val_tn += ((preds == 0) & (labels == 0)).sum().item()

        precision = val_tp / max(val_tp + val_fp, 1)
        recall = val_tp / max(val_tp + val_fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        val_acc = (val_tp + val_tn) / max(val_tp + val_fp + val_fn + val_tn, 1)

        print(f"  Epoch {epoch+1:2d}/{epochs}: "
              f"loss={train_loss/train_total:.3f} "
              f"train_acc={train_correct/train_total:.3f} | "
              f"val_acc={val_acc:.3f} prec={precision:.3f} "
              f"rec={recall:.3f} F1={f1:.3f}")

        if f1 > best_val_f1:
            best_val_f1 = f1
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'f1': f1,
                'precision': precision,
                'recall': recall,
            }, output_path)

    print(f"\nBest model: F1={best_val_f1:.3f}, saved to {output_path}")
    return model


def export_to_onnx(model_path, onnx_path):
    """Export trained model to ONNX for use without PyTorch dependency."""
    if not HAS_TORCH:
        return

    checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
    model = TriangleCNN()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    dummy = torch.randn(1, 1, 64, 64)
    torch.onnx.export(model, dummy, str(onnx_path),
                      input_names=['input'],
                      output_names=['output'],
                      dynamic_axes={'input': {0: 'batch'}})
    print(f"ONNX model exported to {onnx_path}")


if __name__ == '__main__':
    base = Path(__file__).resolve().parent.parent
    data_dir = base / 'training_data'
    model_path = base / 'scripts' / 'triangle_classifier.pth'

    if not HAS_TORCH:
        print("PyTorch not available. Trying to install...")
        import subprocess
        subprocess.check_call([sys.executable, '-m', 'pip', 'install',
                              'torch', 'torchvision', '--quiet'])
        print("Installed PyTorch. Please re-run this script.")
        sys.exit(0)

    # --neg-ratio N : negatives per positive in the training pool
    #                 (0 = use all available); --out PATH : model output
    neg_ratio = 3.0
    if '--neg-ratio' in sys.argv:
        neg_ratio = float(sys.argv[sys.argv.index('--neg-ratio') + 1])
    if '--out' in sys.argv:
        model_path = Path(sys.argv[sys.argv.index('--out') + 1])

    model = train_model(data_dir, model_path, epochs=34, neg_ratio=neg_ratio)

    if model is not None:
        # Export to ONNX
        onnx_path = base / 'scripts' / 'triangle_classifier.onnx'
        try:
            export_to_onnx(model_path, onnx_path)
        except Exception as e:
            print(f"ONNX export failed (non-critical): {e}")
