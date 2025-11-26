# src/train.py
import argparse, os, time
import torch, torch.nn as nn
from torch.optim import Adam
from sklearn.metrics import f1_score
from tqdm import tqdm
from dataset import get_dataloaders
from models import get_resnet50, get_efficientnet_b0
import pandas as pd

def evaluate(model, loader, device):
    model.eval()
    preds, trues = [], []
    probs_list = []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            out = model(imgs)
            prob = torch.softmax(out, dim=1).cpu().numpy()
            pred = prob.argmax(axis=1)
            preds.extend(pred.tolist())
            probs_list.extend(prob.tolist())
            trues.extend(labels.numpy().tolist())
    macro_f1 = f1_score(trues, preds, average='macro')
    return macro_f1, preds, trues, probs_list

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader, test_loader, train_ds, val_ds, test_ds = get_dataloaders(args.data_dir, batch_size=args.batch_size, img_size=args.img_size)
    num_classes = len(train_ds.classes)

    if args.model == 'resnet50':
        model = get_resnet50(num_classes, pretrained=True)
    else:
        model = get_efficientnet_b0(num_classes, pretrained=True)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    best_val = 0.0
    history = []
    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        running_loss = 0.0
        for imgs, labels in pbar:
            imgs = imgs.to(device); labels = labels.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pbar.set_postfix({'loss': running_loss / (pbar.n + 1)})
        # evaluate
        val_f1, _, _, _ = evaluate(model, val_loader, device)
        print(f"[Epoch {epoch+1}] Val Macro-F1: {val_f1:.4f}")
        history.append({'epoch': epoch+1, 'val_macro_f1': val_f1})
        # checkpoint
        ckpt_path = os.path.join(args.out_dir, f"{args.model}_last.pth")
        torch.save({'epoch': epoch+1, 'model_state': model.state_dict()}, ckpt_path)
        if val_f1 > best_val:
            best_val = val_f1
            best_path = os.path.join(args.out_dir, f"{args.model}_best.pth")
            torch.save({'epoch': epoch+1, 'model_state': model.state_dict()}, best_path)
        
        # 🔥 EARLY STOP CLAUSE 🔥
        if val_f1 >= 0.99:
            print(f"\n[Early Stop] Validation Macro-F1 reached {val_f1:.4f} at epoch {epoch+1}. Stopping training early.")
            break


    # final test eval using best
    checkpoint = torch.load(os.path.join(args.out_dir, f"{args.model}_best.pth"))
    model.load_state_dict(checkpoint['model_state'])
    test_f1, preds, trues, probs = evaluate(model, test_loader, device)
    print("Test macro-F1:", test_f1)
    # save probs CSV for ensemble
    df = pd.DataFrame({
        'true': trues,
        'pred': preds,
        'probs': probs  # this will store as lists; you can expand columns if needed
    })
    df.to_csv(os.path.join(args.out_dir, f"{args.model}_test_probs.csv"), index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--model', choices=['resnet50','efficientnet'], default='resnet50')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    train(args)
