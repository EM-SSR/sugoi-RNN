# dnc/ctm_sparse_memory.py
# -----------------------------------------------------------
"""Sparse-Access Memory + CTM per-slot.

旧 CTMMemory (old_memory.py) の機能を
dnc/sparse_memory.py ベースへ移植した実装。
使い方は SparseMemory と同じ。
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from .sparse_memory import SparseMemory
from ctm.ctm import ContinuousThoughtMachine

__all__ = ["CTMSparseMemory"]

class CTMSparseMemory(SparseMemory):
    """スロットごとに CTM を持つ SparseMemory."""

    def __init__(
        self,
        *,
        n_transformer_layers: int = 1,    # = CTM の “思考 tick” 数
        **kwargs
    ):
        # SparseMemory 側で index 構築などを行う
        super().__init__(**kwargs)

        self.n_ticks = max(1, n_transformer_layers)
        heads_ctm    = 1                   # kv_proj / q_proj を有効化
        d_model      = self.cell_size      # ← ＝ memory “word” 次元 D

        # self.mem_size はスロット数 S に対応
        self.ctm = nn.ModuleList([
            ContinuousThoughtMachine(
                # --- 必須引数 ---
                iterations               = self.n_ticks,
                d_model                  = d_model,
                d_input                  = d_model,
                heads                    = heads_ctm,
                n_synch_out              = d_model,
                n_synch_action           = max(1, d_model // 4),
                synapse_depth            = 1,
                memory_length            = 4,
                # --- ここから新たに追加 ---
                deep_nlms                = True,
                memory_hidden_dims       = d_model,
                do_layernorm_nlm         = False,
                # --- 以降はもともとの引数 ---
                backbone_type            = "none",
                positional_embedding_type= "none",
                out_dims                 = d_model,
                prediction_reshaper      = [d_model],
                dropout                  = 0.0,
            )
            for _ in range(self.mem_size)   # スロット数ぶん
        ])

    # ===== 内部 util ==============================================
    def _apply_slot_ctm(self, mem_mat: torch.Tensor) -> torch.Tensor:
        """
        mem_mat : (B, S, D)
        各スロットを CTM に通し，最後の tick 出力 (B,1,D) を連結 → (B,S,D)
        """
        processed = []
        for idx, ctm in enumerate(self.ctm):
            slot_vec = mem_mat[:, idx:idx+1, :]        # (B,1,D)
            preds, _, _ = ctm(slot_vec)               # preds:(B,D,T)
            # 最終の tick 出力だけ取り出す
            processed.append(preds[..., -1].unsqueeze(1))  # (B,1,D)
        return torch.cat(processed, dim=1)               # (B,S,D)

    # ===== public API (旧 Memory と同形) ===========================
    def reset(self, batch_size: int = 1):
        """
        SparseMemory の reset を呼び出し，
        追加で self.mem を保持。
        """
        hidden = super().reset(batch_size)
        # 旧実装と合わせて “可視化用” に直接アクセス出来るメンバも保存
        self.mem = hidden["memory"]
        return hidden

    # ---- READ ----------------------------------------------------
    def read(self, read_w: torch.Tensor) -> torch.Tensor:
        """
        read_w : (B,S,H) → (B,H,D)
        SparseMemory の visible -memory ではなく，
        CTM を通した mem_repr を用いて重み付き和を返す。
        """
        mem_repr = self._apply_slot_ctm(self.mem)           # (B,S,D)
        # read_w は (B, S, H) → (B, H, D)
        return torch.bmm(read_w.transpose(1, 2), mem_repr)  # (B,H,D)

    # ---- WRITE ---------------------------------------------------
    def write(
        self,
        write_w: torch.Tensor,         # (B, S)
        content_vec: torch.Tensor,     # (B, D)  ─ 書き込みキー (= input)
        teacher: torch.Tensor | None = None   # (B, dT) or None
    ) -> torch.Tensor:
        """
        旧 CTMMemory.write と同様，
        ① メモリ行列の上書き  
        ② 対応 CTM へ教師あり更新

        戻り値 : sup_loss (スカラーのテンソル)
        """
        # (1) メモリ / index 更新
        self.mem = self.mem * (1 - write_w.unsqueeze(-1)) \
                 + write_w.unsqueeze(-1) * content_vec.unsqueeze(1)

        # (2) 教師あり更新のための sup_loss を初期化（スカラー）
        sup_loss = torch.zeros((), device=self.mem.device)

        # teacher が与えられている場合のみ CTM に流して sup を足し合わせる
        if teacher is not None:
            # self.ctm は ModuleList や nn.Module のリストで、
            # 各要素 ctm が (preds, certs, sup) = ctm(input, teacher) の戻り値を返す前提
            for idx, ctm in enumerate(self.ctm):
                # idx: CTM のインデックス、self.mem[:, idx:idx+1, :] が (B, 1, D) など
                _, _, sup = ctm(self.mem[:, idx:idx+1, :], teacher)

                # sup は多次元テンソルの可能性があるため、.sum() してスカラーにする
                sup_loss = sup_loss + sup.sum()

        return sup_loss