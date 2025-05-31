# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from dnc.dnc_sam_ctm import DNC
from dnc.controller import BaseController

class CopyDataset(Dataset):
    """
    Copy Task:
     - 入力: [シーケンス, 区切り符号, ゼロパディング]
     - 出力: [ゼロパディング, 区切り符号, シーケンス]
    """
    def __init__(self, seq_len, seq_w, size):
        self.seq_len = seq_len
        self.seq_w   = seq_w
        # ランダムなバイナリ列データ
        self.data    = torch.rand(size, seq_len, seq_w).round()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq       = self.data[idx]                           # (L, W)
        delimiter = torch.ones(1, self.seq_w)                # (1, W)
        pad       = torch.zeros_like(seq)                    # (L, W)

        inp    = torch.cat([seq, delimiter, pad], dim=0)     # (2L+1, W)
        target = torch.cat([pad, delimiter, seq], dim=0)     # (2L+1, W)
        return inp, target

def evaluate(model, loader, device):
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for inp, target in loader:
            inp_t   = inp.transpose(0,1).to(device)          # (T, B, input_size)
            y_hat,_ = model(inp_t)                           # (T, B, output_size)
            pred    = (y_hat.transpose(0,1) > 0.5).float()   # (B, T, W)
            correct += (pred == target.to(device)).all(dim=(1,2)).sum().item()
            total   += inp.shape[0]
    return correct / total

def main():
    # --- ハイパーパラメータ ---
    seq_len     = 10
    seq_w       = 8
    batch_size  = 32
    epochs      = 20
    train_size  = 10_000
    val_size    = 1_000
    device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- データセット・ローダー ---
    train_ds = CopyDataset(seq_len, seq_w, train_size)
    val_ds   = CopyDataset(seq_len, seq_w, val_size)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size)

    # --- モデル定義 ---
    dnc_ext = DNC(
        controller_class     = BaseController,
        input_size           = seq_w,
        output_size          = seq_w,
        mem_slot             = 16,
        mem_size             = 32,
        read_heads           = 4,
        batch_size           = batch_size,
        n_transformer_layers = 2,       # 拡張版: Transformer 層あり
        use_cuda             = device.type=='cuda'
    ).to(device)

    dnc_base = DNC(
        controller_class     = BaseController,
        input_size           = seq_w,
        output_size          = seq_w,
        mem_slot             = 16,
        mem_size             = 32,
        read_heads           = 4,
        batch_size           = batch_size,
        n_transformer_layers = 0,       # ベースライン: Transformer 層なし
        use_cuda             = device.type=='cuda'
    ).to(device)

    # --- オプティマイザ / 損失 ---
    opt_ext  = optim.Adam(dnc_ext.parameters(),  lr=1e-4)
    opt_base = optim.Adam(dnc_base.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    # --- 学習ループ ---
    for ep in range(1, epochs+1):
        dnc_ext.train()
        dnc_base.train()

        train_bar = tqdm(train_loader, desc=f"Epoch {ep}/{epochs} [Train]", leave=False)
        for inp, target in train_bar:
            inp_t  = inp.transpose(0,1).to(device)    # (T, B, W)
            targ_t = target.transpose(0,1).to(device)

            # — 拡張版 DNC —
            opt_ext.zero_grad()
            y_ext, sup_ext = dnc_ext(inp_t)
            loss_ext = criterion(y_ext, targ_t) + sup_ext
            loss_ext.backward()
            torch.nn.utils.clip_grad_norm_(dnc_ext.parameters(), 5.0)
            opt_ext.step()

            # — ベースライン DNC —
            opt_base.zero_grad()
            y_base, sup_base = dnc_base(inp_t)
            loss_base = criterion(y_base, targ_t) + sup_base
            loss_base.backward()
            torch.nn.utils.clip_grad_norm_(dnc_base.parameters(), 5.0)
            opt_base.step()

            # — バッチごとのシーケンス完全一致率計算 —
            with torch.no_grad():
                pred_ext   = (y_ext.transpose(0,1) > 0.5).float()
                acc_ext_b  = (pred_ext == target.to(device)).all(dim=(1,2)).sum().item() / inp.size(0)
                pred_base  = (y_base.transpose(0,1) > 0.5).float()
                acc_base_b = (pred_base == target.to(device)).all(dim=(1,2)).sum().item() / inp.size(0)

            # — プログレスバーに損失と精度を表示 —
            train_bar.set_postfix({
                "L_ext":   f"{loss_ext.item():.3f}",
                "L_base":  f"{loss_base.item():.3f}",
                "A_ext":   f"{acc_ext_b:.2f}",
                "A_base":  f"{acc_base_b:.2f}",
            })

    # --- 最終評価（検証データ）---
    val_bar = tqdm(val_loader, desc="Validating", leave=False)
    final_ext = evaluate(dnc_ext, val_bar, device)
    final_bas = evaluate(dnc_base, val_bar, device)
    print(f"\nFinal Validation Accuracy:")
    print(f"  Extended DNC: {final_ext:.3f}")
    print(f"  Baseline DNC: {final_bas:.3f}")

if __name__ == "__main__":
    main()
