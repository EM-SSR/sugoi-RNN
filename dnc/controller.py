from __future__ import print_function
import torch
from torch.autograd import Variable
import  torch.nn.functional as F
import torch.nn as nn
import numpy as np
import os
import sys

from .util import *



class BaseController(nn.Module):
    def __init__(self, *,
                 input_size: int       = 1,
                 output_size: int      = 1,
                 read_heads: int       = 1,
                 nn_output_size: int   = 64,
                 mem_size: int         = 32,
                 batch_size: int       = 1,
                 recurrent: bool       = True):
        """
        Parameters:
        ----------
        input_size: int
            the size of the data input vector
        output_size: int
            the size of the data output vector
        memory_read_heads: int
            the number of read haeds in the associated external memory
        mem_size: int
            the size of the word in the associated external memory
        batch_size: int
            the size of the input data batch [optional, usually set by the DNC object]
        """
        super().__init__()

        # -------- パラメータ保持 ----------
        self.input_size      = input_size
        self.output_size     = output_size
        self.read_heads      = read_heads
        self.mem_size        = mem_size
        self.nn_output_size  = nn_output_size
        self.recurrent       = recurrent

        # -------- ネットワーク本体 ----------
        self.nn_input_size   = self.mem_size * self.read_heads + self.input_size
        if self.recurrent:
            self.core = nn.GRU(self.nn_input_size, nn_output_size,
                               num_layers=1, batch_first=True)
        else:
            self.core = nn.Sequential(
                nn.Linear(self.nn_input_size, nn_output_size),
                nn.ReLU(inplace=True)
            )

        # -------- 出力線形層（旧実装から流用） ----------
        initrange = 0.1
        self.interface_weights = nn.Parameter(
            torch.empty(nn_output_size,
                        self.mem_size * self.read_heads + 3 * self.mem_size
                        + 5 * self.read_heads + 3).uniform_(-initrange, initrange))
        self.nn_output_weights = nn.Parameter(
            torch.empty(nn_output_size, output_size).uniform_(-initrange, initrange))
        self.mem_output_weights = nn.Parameter(
            torch.empty(self.mem_size * self.read_heads,
                        output_size).uniform_(-initrange, initrange))
                # ── 教師ベクトル生成ヘッド ───────────────
        self.teacher_weights = nn.Parameter(
            torch.empty(self.output_size, self.mem_size).uniform_(-initrange, initrange)
        )


    # ──────────────────────────────────────────────
    def get_state(self, batch_size: int, device=None):
        """
        GRU 用の初期 hidden-state を返す。
        形状 : (num_layers, B, hidden_size) → ここでは (1, B, H)
        """
        if not self.recurrent:
            return None

        if device is None:
            device = next(self.parameters()).device

        # GRU を使う場合は core.hidden_size が正しい
        hidden_size = self.core.hidden_size if isinstance(self.core, nn.GRU) \
                      else self.nn_output_size
        return torch.zeros(1, batch_size, hidden_size, device=device)


    def network_op(self, x, state=None):
        """
        Args:
            x     : (B, in_dim)
            state : GRU hidden state or None
        Returns:
            out   : (B, nn_output_size)
            new_state : 新しい hidden state  (または None)
        """
        if self.recurrent:
            out, new_state = self.core(x.unsqueeze(1), state)  # (B,1,H)
            return out.squeeze(1), new_state
        else:
            return self.core(x), None
                
    def parse_interface_vector(self, interface_vector):
        """
        pasres the flat interface_vector into its various components with their
        correct shapes

        Parameters:
        ----------
        interface_vector: Tensor (batch_size, interface_vector_size)
            the flattened inetrface vector to be parsed

        Returns: dict
            a dictionary with the components of the interface_vector parsed
        """

        parsed = {}
        r_keys_end = self.mem_size * self.read_heads
        r_strengths_end = r_keys_end + self.read_heads
        w_key_end = r_strengths_end + self.mem_size
        erase_end = w_key_end + 1 + self.mem_size
        write_end = erase_end + self.mem_size
        free_end = write_end + self.read_heads

        r_keys_shape = (-1, self.mem_size, self.read_heads)
        r_strengths_shape = (-1, self.read_heads)
        w_key_shape = (-1, self.mem_size, 1)
        write_shape = erase_shape = (-1, self.mem_size)
        free_shape = (-1, self.read_heads)
        modes_shape = (-1, 3, self.read_heads)
        interface_vector = interface_vector.contiguous()
        # parsing the vector into its individual components
        parsed['read_keys'] = interface_vector[:,0:r_keys_end].contiguous().view(*r_keys_shape)
        parsed['read_strengths'] = interface_vector[:, r_keys_end:r_strengths_end].contiguous().view(*r_strengths_shape)
        parsed['write_key'] = interface_vector[:, r_strengths_end:w_key_end].contiguous().view(*w_key_shape)
        parsed['write_strength'] = interface_vector[:, w_key_end].contiguous().view(-1, 1)
        parsed['erase_vector'] = interface_vector[:, w_key_end + 1:erase_end].contiguous().view(*erase_shape)
        parsed['write_vector'] = interface_vector[:, erase_end:write_end].contiguous().view(*write_shape)
        parsed['free_gates'] = interface_vector[:, write_end:free_end].contiguous().view(*free_shape)
        parsed['allocation_gate'] = expand_dims(interface_vector[:, free_end].contiguous(), 1)
        parsed['write_gate'] = expand_dims(interface_vector[:, free_end + 1].contiguous(), 1)
        parsed['read_modes'] = interface_vector[:, free_end + 2:].contiguous().view(*modes_shape)

        # transforming the components to ensure they're in the right ranges
        parsed['read_strengths'] = 1 + F.relu(parsed['read_strengths'])
        parsed['write_strength'] = 1 + F.relu(parsed['write_strength'])
        parsed['erase_vector'] = F.sigmoid(parsed['erase_vector'])
        parsed['free_gates'] =  F.sigmoid(parsed['free_gates'])
        parsed['allocation_gate'] =  F.sigmoid(parsed['allocation_gate'])
        parsed['write_gate'] =  F.sigmoid(parsed['write_gate'])
        parsed['read_modes'] = softmax(parsed['read_modes'], 1)
        
        #for key, value in parsed.iteritems():
        #    value.register_hook(inves('gradient of {}: '.format(key) ))
        #apply_dict(locals())
        return parsed

    def process_input(self, X, last_read_vectors, state=None):
        """
            processes input data through the controller network and returns the
            pre-output and interface_vector

            Parameters:
            ----------
            X: Tensor (batch_size, input_size)
                the input data batch
            last_read_vectors: (batch_size, mem_size, read_heads)
                the last batch of read vectors from memory
            state: Tuple
                state vectors if the network is recurrent

            Returns: Tuple
                pre-output: Tensor (batch_size, output_size)
                parsed_interface_vector: dict
        """

        flat_read_vectors = last_read_vectors.reshape(-1, self.mem_size * self.read_heads)
        complete_input = torch.cat( (X, flat_read_vectors), 1)
        nn_output, nn_state = None, None

        if self.recurrent:
            nn_output, nn_state = self.network_op(complete_input, state)
        else:
            nn_output = self.network_op(complete_input)
                
        pre_output = torch.mm(nn_output, self.nn_output_weights)
        interface = torch.mm(nn_output, self.interface_weights)
        
        #nn_output.register_hook(inves('gradient of nn_output: '))
        #interface.register_hook(inves('gradient of interface: '))
        parsed_interface = self.parse_interface_vector(interface)

        if self.recurrent:
            return pre_output, parsed_interface, nn_state
        else:
            return pre_output, parsed_interface
        
        #apply_dict(locals())
    def final_output(self, pre_output, new_read_vectors):
        """
        returns the final output by taking rececnt memory changes into account

        Parameters:
        ----------
        pre_output: Tensor (batch_size, output_size)
            the ouput vector from the input processing step
        new_read_vectors: Tensor (batch_size, words_size, read_heads)
            the newly read vectors from the updated memory

        Returns: Tensor (batch_size, output_size)
        """

        flat_read_vectors = new_read_vectors.reshape(-1, self.mem_size * self.read_heads)

        final_output = pre_output + torch.mm(flat_read_vectors, self.mem_output_weights)
        
        #apply_dict(locals())
        return final_output

    def forward(self, x, last_read_vectors, state=None):
        """
        x : (B, input_size)
        last_read_vectors : (B, mem_size, read_heads)
        state : (1, B, nn_output_size) | None

        Returns
        -------
        pre_output : (B, output_size)
        write_key  : (B, mem_size)      ― 書き込みキー (= content_vec)
        write_beta : (B,)               ― β（書き込み強度）
        gen_teacher: (B, mem_size)      ― 教師ベクトルを自動生成
        new_state  : GRU hidden state | None
        """
        pre_output, iface, new_state = self.process_input(
            x, last_read_vectors, state
        )

        write_key  = iface["write_key"].squeeze(-1)       # (B, D)
        write_beta = iface["write_strength"].squeeze(-1)  # (B,)

        # 追加：教師ベクトルを生成（tanh で [-1,1] に収める）
        gen_teacher = torch.tanh(torch.mm(pre_output, self.teacher_weights))  # (B, D)

        return pre_output, write_key, write_beta, gen_teacher, new_state
