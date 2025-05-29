import torch
from torch.autograd import Variable
import  torch.nn.functional as F
from copy import copy
import numpy as np
import torch.nn as nn
from collections import namedtuple
from .utils import *
from .memory import Memory, CTMMemory

class DNC(nn.Module):
    """
    * BaseController の引数仕様に合わせて呼び出しを修正。
    * Memory は Transformer 付きに差し替え済み。
    """
    def __init__(self, *,
                 controller_class,
                 input_size: int        = 10,
                 output_size: int       = 10,
                 nn_output_size: int    = 64,
                 mem_slot: int          = 256,
                 mem_size: int          = 64,
                 read_heads: int        = 4,
                 batch_size: int        = 1,
                 n_transformer_layers: int = 0,
                 use_cuda: bool         = True):
        """
        The agent class.

        Parameters:
        -----------
        controller_class: BaseController
            a concrete implementation of the BaseController class
        input_size: int
            the size of the input vector
        output_size: int
            the size of the output vector
        max_sequence_length: int
            the maximum length of an input sequence
        mem_slot: int
            the number of memory slots that can be stored in memory
        mem_size: int
            the size of an individual word in memory
        read_heads: int
            the number of read heads in the memory
        batch_size: int
            the size of the data batch
        """
        super().__init__()
        self.use_cuda = use_cuda
        self.device = torch.device("cuda") if use_cuda is True else torch.device("cpu")

        self.read_heads, self.mem_size = read_heads, mem_size

        # --- Memory (ctm 付き) -------------------------------
        self.memory = CTMMemory(
            mem_slot        = mem_slot,
            mem_size        = mem_size,
            read_heads      = read_heads,
            batch_size      = batch_size,
            n_transformer_layers = n_transformer_layers,
            use_cuda        = use_cuda
        )

        # --- Controller ---------------------------------------------
        self.controller = controller_class(
            input_size=input_size,
            output_size=output_size,
            read_heads=read_heads,
            nn_output_size=nn_output_size,
            mem_size=mem_size,
            batch_size=batch_size,
            recurrent=True      # GRU を使う想定
        )
        
    # ──────────────────────────────────────────────
    def forward(self, xs):
        """
        xs       : (T, B, input_size)
        teachers : None もしくは (T, B, mem_size)
        戻り値   : y_hat (T, B, output_size), sup_loss (スカラー)
        """
        T, B, _ = xs.shape

        # --- 初期化 --------------------------------------------------
        self.memory.reset(B)
        state = self.controller.get_state(B)                     # (1,B,H)
        read_vecs = torch.zeros(                                 # (B,D,H)
            B, self.mem_size, self.read_heads, device=self.device
        )

        outs      = []
        sup_loss  = torch.zeros((), device=self.device)

        for t in range(T):
            # ① コントローラ
            # ① コントローラ
            pre_out, key, beta, gen_teacher, state = self.controller(
                xs[t], read_vecs, state
            )

            # ② 書き込み重み（content addressing）
            sim      = F.cosine_similarity(self.memory.mem, key.unsqueeze(1), dim=-1)  # (B,S)
            write_w  = safe_softmax(beta.unsqueeze(-1) * sim, dim=-1)                  # (B,S)

            # ③ メモリ書き込み（教師あり／自動生成）
            teacher_vec = gen_teacher
            sup_loss    = sup_loss + self.memory.write(write_w, key, teacher_vec)

            # ④ 読み込み
            read_w     = write_w.unsqueeze(-1).repeat(1, 1, self.read_heads)           # (B,S,H)
            read_vecs_ = self.memory.read(read_w)                                      # (B,H,D)
            read_vecs  = read_vecs_.transpose(1, 2)                                    # (B,D,H)

            # ⑤ 出力
            out = self.controller.final_output(pre_out, read_vecs)                     # (B,out)
            outs.append(out)

        return torch.stack(outs), sup_loss
    
    def save(self, ckpts_dir, name):
        raise NotImplementedError
    
    def restore(self, ckpts_dir, name):
        raise NotImplementedError

    #def __call__(self,*args, **kwargs):
    #    return self.forward(*args, **kwargs)


