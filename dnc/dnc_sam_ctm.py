# dnc/dnc_sam_ctm.py
# -----------------------------------------------------------
"""旧 DNC インターフェース互換の SAM + CTM 実装."""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

# 同一パッケージ内の ctm_sparse_memory を参照
from .ctm_sparse_memory import CTMSparseMemory
from .controller import BaseController              # old_controller.py と同一
from .util import safe_softmax                     # old/utils.py

__all__ = ["DNC"]   # そのまま import dnc_sam_ctm as DNC で置き換え可能

class DNC(nn.Module):
    """
    * メモリに CTMSparseMemory を採用
    * forward() の流れは old_dnc.py と同じ
    """

    def __init__(
        self,
        *,
        controller_class,
        input_size             : int  = 10,
        output_size            : int  = 10,
        nn_output_size         : int  = 64,
        mem_slot               : int  = 256,
        mem_size               : int  = 64,
        read_heads             : int  = 4,
        batch_size             : int  = 1,
        n_transformer_layers   : int  = 1,
        use_cuda               : bool = True
    ):
        super().__init__()
        self.device    = torch.device("cuda") if use_cuda else torch.device("cpu")
        self.mem_size  = mem_size
        self.read_heads= read_heads

        # ---- Memory (Sparse + CTM per-slot) -----------------------
        self.memory = CTMSparseMemory(
            input_size             = mem_size,           # 不使用だがダミーで渡す
            mem_size               = mem_slot,           # スロット数 S
            cell_size              = mem_size,           # ワード次元 D
            read_heads             = read_heads,
            sparse_reads           = read_heads,         # “可視”セル数 = read_heads
            n_transformer_layers   = n_transformer_layers,
            device                 = self.device,
        )

        # ---- Controller ------------------------------------------
        self.controller = controller_class(
            input_size            = input_size,
            output_size           = output_size,
            read_heads            = read_heads,
            nn_output_size        = nn_output_size,
            mem_size              = mem_size,
            batch_size            = batch_size,
            recurrent             = True,            # GRU ベース
        )

    # ===============================================================
    def forward(self, xs: torch.Tensor):
        """
        xs : (T,B,input_size)
        戻り値 :
          y_hat   : (T,B,output_size)
          sup_loss: スカラー（CTM への教師損失）
        """
        T, B, _ = xs.shape
        # (1) Memory の hidden 初期化
        self.memory.reset(B)
        # (2) Controller の初期 hidden state を取得
        state     = self.controller.get_state(B)
        # (3) read_vecs: (B, D, H) のゼロテンソルを用意 (D=mem_size, H=read_heads)
        read_vecs = torch.zeros(
            B, self.mem_size, self.read_heads, device=self.device
        )

        outs, sup_loss = [], torch.zeros((), device=self.device)

        for t in range(T):
            # (1) Controller ------------------------------------------------
            # BaseController は以下を返す想定:
            # pre_out: Controller の出力 (B, nn_output_size)
            # key:      書き込みキーとして使う (B, mem_size)
            # beta:     強度スカラー (B,)
            # gen_teacher: CTM に流す教師ベクトル (B, dT) or None
            # state:    次の GRU hidden state
            pre_out, key, beta, gen_teacher, state = self.controller(
                xs[t], read_vecs, state
            )

            # (2) 書き込み重み（content addressing）------------------------
            sim     = F.cosine_similarity(self.memory.mem, key.unsqueeze(1), dim=-1)  # (B, S)
            write_w = safe_softmax(beta.unsqueeze(-1) * sim, dim=-1)                  # (B, S)

            # (3) Memory WRITE + CTM supervise --------------------------------
            sup_loss = sup_loss + self.memory.write(write_w, key, gen_teacher)

            # (4) Memory READ --------------------------------------------------
            read_w     = write_w.unsqueeze(-1).repeat(1, 1, self.read_heads)         # (B, S, H)
            read_vecs  = self.memory.read(read_w)                                    # (B, H, D)

            # (5) Controller 出力生成 (省略例)
            # 例: pre_out と read_vecs を結合し、y_hat を生成するとする
            concat = torch.cat([pre_out, read_vecs.view(B, -1)], dim=-1)             # (B, nn_output + H*D)
            y_hat  = pre_out                                                        # (省略: 実際は線形層などで output_size に投影)
            outs.append(y_hat)

        # リストをテンソル (T, B, output_size) にまとめて返す
        y_all = torch.stack(outs, dim=0)
        return y_all, sup_loss
