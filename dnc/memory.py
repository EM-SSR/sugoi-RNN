import torch
from copy import deepcopy
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import TransformerEncoderLayer, TransformerEncoder
import matplotlib.pyplot as plt
from collections import namedtuple

import numpy as np
from collections import namedtuple
from .utils import *
from ctm.ctm import ContinuousThoughtMachine

mem_tuple = namedtuple(
    'mem_tuple',
    'mem_mat, mem_usage, pre_vec, link_mat, write_weight, read_weight, read_vec'
)

class Memory(nn.Module):
    """
        Parameters:
        ----------
        mem_slot: int
            the maximum number of words that can be stored in the memory at the
            same time
        mem_size: int
            the size of the individual word in the memory
        read_heads: int
            the number of read heads that can read simultaneously from the memory
        batch_size: int
            the size of input data batch
    """
    def __init__(self, mem_slot=256, mem_size=64, read_heads=4, batch_size=1,
                 n_transformer_layers=0, use_cuda=True):
        super().__init__()
        self.S, self.D = mem_slot, mem_size
        self.H, self.B = read_heads, batch_size
        self.mem_slot, self.mem_size = mem_slot, mem_size
        self.read_heads, self.batch_size = read_heads, batch_size
        self.use_tf   = n_transformer_layers > 0
        self.device   = 'cuda' if use_cuda else 'cpu'
        self.use_cuda = use_cuda

        # ── Transformer per slot ───────────────────────────────────
        if self.use_tf:
            base = TransformerEncoderLayer(d_model=self.D,
                                           nhead=max(1, self.D // 8),
                                           batch_first=True)
            self.tf = nn.ModuleList([
                TransformerEncoder(deepcopy(base), num_layers=n_transformer_layers)
                for _ in range(self.S)
            ]).to(self.device)

        # ── util for link-matrix self-mask ──────────────────────────
        self.I = Variable(expand_dims(torch.eye(self.S), 0))      # 1,S,S
        if self.use_cuda:
            self.I = self.I.cuda()

        # ── namedtuple for state -----------------------------------
        self.mem_tuple = mem_tuple

        self.init_memory(batch_size)

    def _apply_cell_transformers(self, mem_mat):
        """
        mem_mat : (B, S, D)
        各スロットごとに専用Transformerを適用して再結合
        """
        if not self.use_tf:
            return mem_mat
        out_list = []
        for idx, enc in enumerate(self.tf):
            # スロットごとに切り出し
            tok = mem_mat[:, idx:idx+1, :]   # (B,1,D)
            out = enc(tok)                  # (B,1,D)
            out_list.append(out)
        # (B,S,D) に戻す
        return torch.cat(out_list, dim=1)

    def update_read_vectors(self, memory_matrix, read_weights):
        """
        メモリ => Transformer（任意）=> 重み付き和 で read vector を返す
        """
        mem_repr = self._apply_cell_transformers(memory_matrix)   # (B,S,D)
        return torch.bmm(mem_repr.transpose(1, 2), read_weights)  # (B,D,H)

    def init_memory(self, batch_size):
        """Return a fresh memory state & keep internal copy for可視化."""
        mem_list = [
            Variable(torch.zeros(batch_size, self.S, self.D).fill_(1e-6), requires_grad=True),
            Variable(torch.zeros(batch_size, self.S),                     requires_grad=True),
            Variable(torch.zeros(batch_size, self.S),                     requires_grad=True),
            Variable(torch.zeros(batch_size, self.S, self.S),             requires_grad=True),
            Variable(torch.zeros(batch_size, self.S).fill_(1e-6),         requires_grad=True),
            Variable(torch.zeros(batch_size, self.S, self.H).fill_(1e-6), requires_grad=True),
            Variable(torch.zeros(batch_size, self.D, self.H).fill_(1e-6), requires_grad=True)
        ]
        if self.use_cuda:
            mem_list = [m.cuda() for m in mem_list]

        self.mem_mat = mem_list[0]      # 可視化用に保存
        return self.mem_tuple._make(mem_list)

        
    def get_content_address(self, memory_matrix, query_keys, strengths):
        """
        retrives a content-based adderssing weights given the keys

        Parameters:
        ----------
        memory_matrix: Tensor (batch_size, mem_slot, mem_size)
            the memory matrix to lookup in
        query_keys: Tensor (batch_size, mem_size, number_of_keys)
            the keys to query the memory with
        strengths: Tensor (batch_size, number_of_keys, )
            the list of strengths for each lookup query_keys
        
        Returns: Tensor (batch_size, mem_slot, number_of_keys)
            The list of lookup weightings for each provided key
        """
        # cos_dist is (batch_size, mem_slot, number_of_keys)
        cos_dist = cosine_distance(memory_matrix, query_keys)
        
        strengths = expand_dims(strengths, 1).expand_as(cos_dist)
        #apply_dict(locals())
        return softmax(cos_dist*strengths, 1)

    def update_usage_vector(self, usage_vector, read_weights, write_weight, free_gates):
        """
        updates and returns the usgae vector given the values of the free gates
        and the usage_vector, read_weights, write_weight from previous step

        Parameters:
        ----------
        usage_vector: Tensor (batch_size, mem_slot)
        read_weights: Tensor (batch_size, mem_slot, read_heads)
        write_weight: Tensor (batch_size, mem_slot)
        free_gates: Tensor (batch_size, read_heads, )

        Returns: Tensor (batch_size, mem_slot, )
            the updated usage vector
        """
        free_gates = expand_dims(free_gates,1).expand_as(read_weights)
        retention_vector = torch.prod(2- read_weights * free_gates, 2)
        updated_usage = (usage_vector + write_weight - usage_vector * write_weight)  * retention_vector
        #apply_dict(locals())                
        return updated_usage

    def get_allocation_weight(self, sorted_usage, free_list):
        """
        retreives the writing allocation weight based on the usage free list

        Parameters:
        ----------
        sorted_usage: Tensor (batch_size, mem_slot, )
            the usage vector sorted ascendly
        free_list: Tensor (batch, mem_slot, )
            the original indecies of the sorted usage vector

        Returns: Tensor (batch_size, mem_slot, )
            the allocation weight for each word in memory
        """

        #shifted_cumprod =  cumprod(sorted_usage, dim = 1) / (sorted_usage[0] + 1e-8)
        #shifted_cumprod[-1] = shifted_cumprod[-1]/(sorted_usage[-1]+1e-8)
        batch_size = free_list.size()[0]
        shifted_cumprod =  cumprod(sorted_usage, dim = 1, exclusive = True)
        #shifted_cumprod = sorted_usage
        unordered_allocation_weight = (1 - sorted_usage) * shifted_cumprod

        index_mapper = Variable(
            torch.from_numpy(np.cumsum([0] + [self.mem_slot] * (batch_size - 1))[:, np.newaxis]),  requires_grad = False
        ).expand(batch_size, self.mem_slot)
        
        #index_mapper = index_mapper.cuda(free_list.get_device()) if free_list.is_cuda else index_mapper
        index_mapper = to_device(index_mapper, free_list)
        mapped_free_list = free_list + index_mapper
        flat_unordered_allocation_weight = unordered_allocation_weight.view(-1)
        flat_mapped_free_list = mapped_free_list.view(-1)
        flat_container = Variable(sorted_usage.data.new(self.batch_size * self.mem_slot).fill_(0), requires_grad= False)

        #flat_ordered_weights = flat_container.scatter(
        #    flat_mapped_free_list,
        #    flat_unordered_allocation_weight
        #)
        flat_ordered_weights = flat_container.scatter_(0,
            flat_mapped_free_list.data,
            flat_unordered_allocation_weight.data
        )
        #flat_ordered_weights = to_device(flat_ordered_weights, free_list)
        #apply_dict(locals())
        return flat_ordered_weights.view(self.batch_size, self.mem_slot)

    def update_write_weight(self, lookup_weight, allocation_weight, write_gate, allocation_gate):
        """
        updates and returns the current write_weight

        Parameters:
        ----------
        lookup_weight: Tensor (batch_size, mem_slot, number_of_keys)
            the weight of the lookup operation in writing
        allocation_weight: Tensor (batch_size, mem_slot)
            the weight of the allocation operation in writing
        write_gate: (batch_size, 1)
            the fraction of writing to be done
        allocation_gate: (batch_size, 1)
            the fraction of allocation to be done

        Returns: Tensor (batch_size, mem_slot)
            the updated write_weight
        """

        # remove the dimension of 1 from the lookup_weight
        first_2_size = lookup_weight.size()[0:2]
        lookup_weight = lookup_weight.view(*first_2_size)
        alloc_wshape = allocation_weight.size()
        
        expand_ag = allocation_gate.expand(*alloc_wshape)
        updated_write_weight = write_gate.expand(*alloc_wshape) * ( expand_ag * \
                               allocation_weight + (1 - expand_ag) * lookup_weight)
        
        #apply_dict(locals())
        
        return updated_write_weight

    def update_memory(self, memory_matrix, write_weight, write_vector, erase_vector):
        """
        updates and returns the memory matrix given the weight, write and erase vectors
        and the memory matrix from previous step

        Parameters:
        ----------
        memory_matrix: Tensor (batch_size, mem_slot, mem_size)
            the memory matrix from previous step
        write_weight: Tensor (batch_size, mem_slot)
            the weight of writing at each memory location
        write_vector: Tensor (batch_size, mem_size)
            a vector specifying what to write
        erase_vector: Tensor (batch_size, mem_size)
            a vector specifying what to erase from memory

        Returns: Tensor (batch_size, mem_slot, mem_size)
            the updated memory matrix
        """

        # expand data with a dimension of 1 at multiplication-adjacent location
        # to force matmul to behave as an outer product
        write_weight = expand_dims(write_weight, 2)
        write_vector = expand_dims(write_vector, 1)
        erase_vector = expand_dims(erase_vector, 1)

        erasing = memory_matrix * (1 - torch.bmm(write_weight, erase_vector))
        writing = torch.bmm(write_weight, write_vector)
        updated_memory = erasing + writing
        
        #apply_dict(locals())
        return updated_memory

    def update_precedence_vector(self, precedence_vector, write_weight):
        """
        updates the precedence vector given the latest write weight
        and the precedence_vector from last step

        Parameters:
        ----------
        precedence_vector: Tensor (batch_size. mem_slot)
            the precedence vector from the last time step
        write_weight: Tensor (batch_size,mem_slot)
            the latest write weight for the memory

        Returns: Tensor (batch_size, mem_slot)
            the updated precedence vector
        """

        reset_factor = 1 - reduce_sum(write_weight, 1, keep_dim=True)
        updated_precedence_vector = reset_factor.expand_as(precedence_vector) * \
                                    precedence_vector + write_weight
        #apply_dict(locals())                                                
        return updated_precedence_vector
    
    def update_link_matrix(self, precedence_vector, link_matrix, write_weight):
        """
        updates and returns the temporal link matrix for the latest write
        given the precedence vector and the link matrix from previous step

        Parameters:
        ----------
        precedence_vector: Tensor (batch_size, mem_slot)
            the precedence vector from the last time step
        link_matrix: Tensor (batch_size, mem_slot, mem_slot)
            the link matrix form the last step
        write_weight: Tensor (batch_size, mem_slot)
            the latest write_weight for the memory

        Returns: Tensor (batch_size, mem_slot, mem_slot)
            the updated temporal link matrix
        """

        
        precedence_vector = expand_dims(precedence_vector, 1)

        reset_factor = 1 - pairwise_add(write_weight, is_batch=True)
        
        write_weight = expand_dims(write_weight, -1)

        updated_link_matrix = reset_factor * link_matrix + torch.bmm(write_weight, precedence_vector)
        updated_link_matrix = (1 - self.I).expand_as(updated_link_matrix) * updated_link_matrix  # eliminates self-links
        
        #apply_dict(locals())
        
        return updated_link_matrix
    
    def get_directional_weights(self, read_weights, link_matrix):
        """
        computes and returns the forward and backward reading weights
        given the read_weights from the previous step

        Parameters:
        ----------
        read_weights: Tensor (batch_size, mem_slot, read_heads)
            the read weights from the last time step
        link_matrix: Tensor (batch_size, mem_slot, mem_slot)
            the temporal link matrix

        Returns: Tuple
            forward weight: Tensor (batch_size, mem_slot, read_heads),
            backward weight: Tensor (batch_size, mem_slot, read_heads)
        """

        forward_weight = torch.bmm(link_matrix, read_weights)
        backward_weight = torch.bmm(link_matrix.transpose(1,2), read_weights)

        #apply_dict(locals())
        return forward_weight, backward_weight

    def update_read_weights(self, lookup_weights, forward_weight, backward_weight, read_mode):
        """
        updates and returns the current read_weights

        Parameters:
        ----------
        lookup_weights: Tensor (batch_size, mem_slot, read_heads)
            the content-based read weight
        forward_weight: Tensor (batch_size, mem_slot, read_heads)
            the forward direction read weight
        backward_weight: Tensor (batch_size, mem_slot, read_heads)
            the backward direction read weight
        read_mode: Tesnor (batch_size, 3, read_heads)
            the softmax distribution between the three read modes

        Returns: Tensor (batch_size, mem_slot, read_heads)
        """

        backward_mode = expand_dims(read_mode[:, 0, :].contiguous(), 1).expand_as(backward_weight) * backward_weight
        lookup_mode   = expand_dims(read_mode[:, 1, :].contiguous(), 1).expand_as(lookup_weights)  * lookup_weights
        forward_mode  = expand_dims(read_mode[:, 2, :].contiguous(), 1).expand_as(forward_weight)  * forward_weight
        updated_read_weights = backward_mode + lookup_mode + forward_mode
        
        #apply_dict(locals())
        return updated_read_weights

    # ---------------------------------------------------------------
    def reset(self, batch):
        self.mem = torch.zeros(batch, self.S, self.D, device=self.device) + 1e-6
        self.usage = torch.zeros(batch, self.S, device=self.device)

    # ---------------------------------------------------------------
    def _apply_tf(self, mem_mat):
        if not self.use_tf:
            return mem_mat
        out = []
        for i, enc in enumerate(self.tf):
            out.append(enc(mem_mat[:, i:i+1, :]))      # (B,1,D)
        return torch.cat(out, 1)                       # (B,S,D)

    # === READ ======================================================
    def read(self, read_w):
        """read_w : (B,S,H) → read_vecs : (B,H,D)"""
        mem_repr = self._apply_tf(self.mem)            # (B,S,D)
        return torch.bmm(read_w.transpose(1, 2), mem_repr)

    # === WRITE（教師信号込み）=====================================
    def write(self, write_w, content_vec, teacher=None):
        """
        Parameters
        ----------
        write_w      : (B, S)    ― 書き込み重み
        content_vec  : (B, D)    ― 書き込む入力ベクトル
        teacher      : (B, dT) or None
                       教師ベクトル（次元 dT は D と異なっていても OK）

        Returns
        -------
        sup_loss : torch scalar   ― 教師信号がある場合の損失
        """
        # ---- ① メモリ本体への書き込み -----------------------------
        self.mem = self.mem * (1 - write_w.unsqueeze(-1)) \
                 + write_w.unsqueeze(-1) * content_vec.unsqueeze(1)

        # ---- ② スロット Transformer への教師あり学習 -------------
        sup_loss = torch.zeros((), device=self.device)
        if teacher is not None and self.use_tf:
            B = content_vec.size(0)
            # 書き込み重みが最大のスロットだけを選択（＝その Transformer だけ勾配が流れる）
            chosen = write_w.argmax(dim=1)   # (B,)

            for b in range(B):
                slot = chosen[b].item()      # 0-based index

                # 教師ベクトルを D 次元に合わせてパディング／切り詰め
                tgt = teacher[b]
                if tgt.shape[-1] != self.D:
                    if tgt.shape[-1] < self.D:
                        tgt = F.pad(tgt, (0, self.D - tgt.shape[-1]))
                    else:
                        tgt = tgt[..., : self.D]

                # 2 トークン（入力・教師）を 1 シークエンスとして Transformer へ
                seq = torch.stack([content_vec[b], tgt], dim=0).unsqueeze(0)  # (1,2,D)
                pred_seq = self.tf[slot](seq)                                 # (1,2,D)

                # 先頭トークンの出力を教師ベクトルと照合
                pred = pred_seq[:, 0, :]        # (1,D)
                sup_loss = sup_loss + F.mse_loss(pred.squeeze(0), tgt)

        return sup_loss

    # === 可視化 ====================================================
    def visualize(self, batch_id: int = 0, slots=None,
                  save_path: str | None = None, show: bool = True):
        """
        batch_id : 何番目のバッチを可視化するか
        slots    : None なら全スロット，リストならその番号だけ
        save_path: 画像を保存したい場合のパス
        show     : True なら plt.show() も実行

        返値 : matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt

        if batch_id >= self.mem.size(0):
            raise ValueError("batch_id out of range")

        mem_cpu = self.mem.detach().cpu()[batch_id]  # (S,D)
        if slots is not None:
            mem_cpu = mem_cpu[slots]

        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.imshow(mem_cpu, vmin=-1, vmax=1, aspect='auto', cmap='viridis')
        ax.set_ylabel('Memory slot')
        ax.set_xlabel('Feature dim')
        fig.colorbar(im, ax=ax, label='value')

        if save_path is not None:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')

        if show:
            plt.show()
        plt.close(fig)
        return fig

class CTMMemory(Memory):
    """
    Memory ↔ ContinuousThoughtMachine 組み合わせ版
    （TransformerEncoder を CTM へ置換）
    """

    def __init__(self, *, n_transformer_layers: int = 1, **kwargs):
        # 親 Memory は transformer を使わないので層数=0 で初期化
        super().__init__(n_transformer_layers=0, **kwargs)

        # ---- CTM をスロット数ぶん用意 ------------------------------
        heads_ctm = 1                # ★ heads>0 にして kv_proj / q_proj を有効化
        self.ctm = nn.ModuleList([
            ContinuousThoughtMachine(
                iterations          = max(1, n_transformer_layers),  # “思考 tick” 数
                d_model             = self.mem_size,
                d_input             = self.mem_size,
                heads               = heads_ctm,
                n_synch_out         = self.mem_size,
                n_synch_action      = max(1, self.mem_size // 4),
                synapse_depth       = 1,
                memory_length       = 4,
                deep_nlms           = False,
                memory_hidden_dims  = self.mem_size,
                do_layernorm_nlm    = False,
                backbone_type       = 'none',
                positional_embedding_type = 'none',
                out_dims            = self.mem_size,
                prediction_reshaper = [self.mem_size],
                dropout             = 0.0,
                neuron_select_type  = 'random-pairing',
                n_random_pairing_self = 0
            )
            for _ in range(self.mem_slot)
        ])

    # ---------- スロットごとに CTM “思考” を適用 ----------------------
    def _apply_cell_transformers(self, mem_mat: torch.Tensor) -> torch.Tensor:
        """
        mem_mat : (B, S, D)
        各スロットを CTM に通し、最後の internal tick の prediction を返す。
        """
        processed = []
        for idx, ctm in enumerate(self.ctm):
            slot_vec = mem_mat[:, idx:idx+1, :]                 # (B,1,D)
            preds, _, _ = ctm(slot_vec)                         # preds:(B,D,T)
            processed.append(preds[..., -1].unsqueeze(1))       # (B,1,D)
        return torch.cat(processed, dim=1)                      # (B,S,D)

    # ---------- 読み取り ---------------------------------------------
    def read(self, read_w: torch.Tensor) -> torch.Tensor:
        """
        read_w : (B,S,H) → (B,H,D)
        """
        mem_repr = self._apply_cell_transformers(self.mem)      # (B,S,D)
        return torch.bmm(read_w.transpose(1, 2), mem_repr)

    # ---------- 書き込み（CTM へ教師誤差も伝搬）----------------------
    def _ctm_supervise(self, write_w, teacher):
        """
        書き込んだスロットに対して CTM を微調整
        """
        B = write_w.size(0)
        loss = torch.zeros((), device=self.device)

        chosen = write_w.argmax(dim=1)                          # (B,)
        for b in range(B):
            slot = chosen[b].item()
            tgt  = teacher[b]

            # tgt 次元を D に合わせてパディング／トリム
            if tgt.shape[-1] != self.mem_size:
                if tgt.shape[-1] < self.mem_size:
                    tgt = F.pad(tgt, (0, self.mem_size - tgt.shape[-1]))
                else:
                    tgt = tgt[..., :self.mem_size]

            preds, _, _ = self.ctm[slot](self.mem[b:b+1, slot:slot+1, :])
            pred_vec = preds[0, :, -1]                          # (D,)
            loss = loss + F.mse_loss(pred_vec, tgt)

        return loss

    def write(self, write_w, content_vec, teacher=None):
        """
        標準の書き込みに加えて CTM 用の sup_loss を加算
        """
        sup_loss = super().write(write_w, content_vec, teacher=None)
        if teacher is not None:
            sup_loss = sup_loss + self._ctm_supervise(write_w, teacher)
        return sup_loss