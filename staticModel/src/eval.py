"""
eval.py
---------
Evaluates a trained malware image classification model (ResNet50 or EfficientNet)
on the test dataset, computes metrics, and saves results.

Usage Example:
    python src/eval.py --data_dir ./data --model_path ./models/resnet50_best.pth --model resnet50 --out_dir ./results/resnet_eval
"""

import os
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from dataset import get_dataloaders
from models import get_resnet50, get_efficientnet_b0
from seed import set_seed

# ===================== Evaluate Function =====================
def evaluate_model(model, loader, device):
    model.eval()
    all_preds, all_probs, all_labels = [], [], []

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Evaluating", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)
            preds = probs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_preds), np.array(all_probs), np.array(all_labels)


# ===================== Confusion Matrix Plot =====================
def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# ===================== Main Evaluation Script =====================
def main(args):
    set_seed(42)

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    _, _, test_loader, train_ds, _, test_ds = get_dataloaders(args.data_dir, batch_size=args.batch_size, img_size=args.img_size)
    class_names = test_ds.classes
    num_classes = len(class_names)

    # Load model
    if args.model == "resnet50":
        model = get_resnet50(num_classes=num_classes, pretrained=False)
    else:
        model = get_efficientnet_b0(num_classes=num_classes, pretrained=False)

    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    model = model.to(device)

    print(f"[INFO] Loaded model from: {args.model_path}")

    # Evaluate
    preds, probs, labels = evaluate_model(model, test_loader, device)

    # Compute metrics
    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='macro', zero_division=0)
    recall = recall_score(labels, preds, average='macro', zero_division=0)
    f1 = f1_score(labels, preds, average='macro', zero_division=0)

    # Print & Save results
    print("\n===== Evaluation Metrics =====")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")

    print("\n===== Classification Report =====")
    report = classification_report(labels, preds, target_names=class_names, digits=4)
    print(report)

    # Save metrics and report
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1'],
        'Value': [acc, precision, recall, f1]
    })
    metrics_df.to_csv(os.path.join(args.out_dir, "metrics.csv"), index=False)

    with open(os.path.join(args.out_dir, "classification_report.txt"), 'w') as f:
        f.write(report)

    # Save predictions
    df = pd.DataFrame(probs, columns=[f"class_{c}" for c in class_names])
    df.insert(0, 'file_id', [test_ds.imgs[i][0] for i in range(len(test_ds))])  # optional, path or filename
    df['predicted_family'] = [class_names[i] for i in preds]
    df['family_confidence'] = probs.max(axis=1)
    df['static_mal_prob'] = 1 - df['class_benign'] if 'class_benign' in df.columns else df['family_confidence']
    df.to_csv(os.path.join(args.out_dir, "predictions.csv"), index=False)

    # Save confusion matrix plot
    plot_confusion_matrix(labels, preds, class_names, os.path.join(args.out_dir, "confusion_matrix.png"))

    print(f"\n[+] Results saved in: {args.out_dir}")


# ===================== CLI Entry =====================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate trained malware image classification model.")
    parser.add_argument("--data_dir", required=True, help="Path to dataset directory (with train/val/test).")
    parser.add_argument("--model_path", required=True, help="Path to trained model checkpoint (.pth).")
    parser.add_argument("--out_dir", required=True, help="Directory to save evaluation results.")
    parser.add_argument("--model", choices=["resnet50", "efficientnet"], default="resnet50", help="Model architecture.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation.")
    parser.add_argument("--img_size", type=int, default=224, help="Image size for evaluation.")
    args = parser.parse_args()

    main(args)
