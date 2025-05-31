# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import copy
from torch.autograd import Variable
from typing import Callable

# ----------------------------------------------------------------------
# 勾配が暴走しやすい箇所での NaN 対策として ε を追加
# ----------------------------------------------------------------------
EPS = 1e-8                              # NaN 防止用の極小値

δ = 1e-6


def recursiveTrace(obj: torch.Tensor | torch.nn.Module | None) -> None:
    """Recursively traces the computational graph of a tensor or module.

    Args:
        obj: The tensor or module to trace.
    """
    if obj is None:
        return

    print(type(obj))
    if hasattr(obj, "grad_fn"):
        print(obj.grad_fn)
        recursiveTrace(obj.grad_fn)  # type: ignore
    elif hasattr(obj, "next_functions"):
        print(obj.requires_grad, len(obj.next_functions))  # type: ignore
        for f, _ in obj.next_functions:  # type: ignore
            recursiveTrace(f)


def cuda(x: torch.Tensor, requires_grad: bool = False, device: torch.device | None = None) -> torch.Tensor:
    """Moves a tensor to the specified device (CPU or GPU).

    Args:
        x: The tensor to move.
        requires_grad: Whether the tensor should require gradients.
        device: The device to move the tensor to.  Defaults to CPU.

    Returns:
        The tensor on the specified device.
    """
    if device is None:
        return x.float().requires_grad_(requires_grad)
    else:
        return x.float().to(device).requires_grad_(requires_grad)


def cudavec(x: np.ndarray, requires_grad: bool = False, device: torch.device | None = None) -> torch.Tensor:
    """Creates a tensor from a NumPy array and moves it to the specified device.

    Args:
        x: The NumPy array.
        requires_grad: Whether the tensor should require gradients.
        device: The device. Defaults to cpu.

    Returns:
        The tensor on the specified device.
    """
    return cuda(torch.Tensor(x), requires_grad, device)


def cudalong(x: np.ndarray, requires_grad: bool = False, device: torch.device | None = None) -> torch.Tensor:
    """Creates a LongTensor from a NumPy array and moves it to the specified device.

    Args:
        x: The NumPy array.
        requires_grad: Whether the tensor should require gradients.
        device: The device. Defaults to CPU

    Returns:
        The LongTensor on the specified device.
    """
    return cuda(torch.LongTensor(x.astype(np.int64)), requires_grad, device)


def θ(a: torch.Tensor, b: torch.Tensor, norm_by: int = 2) -> torch.Tensor:
    """Calculates the batchwise cosine similarity between two tensors.

    Args:
        a: A 3D tensor (b * m * w).
        b: A 3D tensor (b * r * w).
        norm_by: The norm to use for normalization.

    Returns:
        The batchwise cosine similarity (b * r * m).
    """
    dot = torch.bmm(a, b.transpose(1, 2))
    a_norm = torch.norm(a, p=norm_by, dim=2).unsqueeze(2)
    b_norm = torch.norm(b, p=norm_by, dim=2).unsqueeze(1)
    cos = dot / (a_norm * b_norm + δ)
    return cos.transpose(1, 2).contiguous()


def σ(input: torch.Tensor, axis: int = 1) -> torch.Tensor:  # NOQA
    """Applies the softmax function along a specified axis.

    Args:
        input: The input tensor.
        axis: The axis along which to apply softmax.

    Returns:
        The softmax output.
    """
    return F.softmax(input, dim=axis)


def register_nan_checks(model: nn.Module) -> None:
    """Registers backward hooks to check for NaN gradients.

    Args:
        model: The model to register hooks on.
    """

    def check_grad(
        module: nn.Module, grad_input: tuple[torch.Tensor | None, ...], grad_output: tuple[torch.Tensor | None, ...]
    ) -> None:
        if any(torch.isnan(gi).any() for gi in grad_input if gi is not None):
            print(f"NaN gradient in grad_input of {type(module).__name__}")

    for module in model.modules():
        module.register_full_backward_hook(check_grad)  # type: ignore


def apply_dict(dic: dict) -> None:
    """Applies gradient NaN checks to a dictionary of variables.

    Args:
        dic: The dictionary.
    """
    for k, v in dic.items():
        apply_var(v, k)
        if isinstance(v, nn.Module):
            key_list = [a for a in dir(v) if not a.startswith("__")]
            for key in key_list:
                apply_var(getattr(v, key), key)
            for pk, pv in v.named_parameters():
                apply_var(pv, pk)


def apply_var(v: torch.Tensor | nn.Module | None, k: str) -> None:
    """Applies gradient NaN checks to a variable.

    Args:
        v: The variable.
        k: The name of the variable.
    """
    if isinstance(v, torch.Tensor) and v.requires_grad:
        v.register_hook(check_nan_gradient(k))


def check_nan_gradient(name: str = "") -> Callable[[torch.Tensor], torch.Tensor | None]:
    """Creates a hook to check for NaN gradients.

    Args:
        name: The name of the variable.

    Returns:
        The hook function.
    """

    def f(tensor: torch.Tensor) -> torch.Tensor | None:
        if torch.isnan(tensor).any():
            print(f"\nnan gradient of {name}:")
            return tensor
        return None

    return f


def ptr(tensor: torch.Tensor) -> int:
    """Returns the memory address of a tensor.

    Args:
        tensor: The tensor.

    Returns:
        The memory address.
    """
    return tensor.data_ptr()


def ensure_gpu(tensor: torch.Tensor | np.ndarray, device: torch.device | None) -> torch.Tensor:
    """Ensures a tensor is on the specified GPU.

    Args:
        tensor: The tensor or NumPy array.
        device: The device

    Returns:
        The tensor on the specified GPU.
    """
    if isinstance(tensor, torch.Tensor) and device is not None:
        return tensor.to(device)
    elif isinstance(tensor, np.ndarray) and device is not None:
        return torch.tensor(tensor, device=device)
    elif isinstance(tensor, np.ndarray):
        return torch.Tensor(tensor)
    else:
        return tensor


def print_gradient(x: torch.Tensor, name: str) -> None:
    """Prints the gradient of a tensor.

    Args:
        x: The tensor.
        name: name of tensor
    """
    s = "Gradient of " + name + " ----------------------------------"
    x.register_hook(lambda y: print(s, y.squeeze()))

# ----------------------------------------------------------------------

def safe_softmax(x, dim=-1):
    x = x - x.max(dim=dim, keepdim=True).values
    return F.softmax(x, dim=dim)

def safe_norm(x, p=2, dim=-1, keepdim=False):
    return torch.clamp(x.norm(p=p, dim=dim, keepdim=keepdim), min=EPS)

# === 既存 API の置き換え（警告解消） ===============================
def softmax(input_, axis=1):
    return safe_softmax(input_, dim=axis)

def softmax_2d(input_2d, dim=-1):
    return safe_softmax(input_2d, dim=dim)

def apply_var(v, k):
    if isinstance(v, Variable) and v.requires_grad:
            v.register_hook(inves(k))
            
def apply_dict(dic):
    for k, v in dic.iteritems():
        apply_var(v, k)
        if isinstance(v, nn.Module):
           key_list = [a for a in dir(v) if not a.startswith('__')]
           for key in key_list:
               apply_var(getattr(v, key), key)
           for pk, pv in v._parameters.iteritems():
               apply_var(pv, pk)
        
def inves(name=''):
    def f(tensor):
        if np.isnan(torch.mean(tensor).data.cpu().numpy() ):
            print('\ngradient of {} :'.format(name))
            print(tensor)
            assert 0, 'nan gradient'
            return tensor
    return f
        
def reduce_sum(inputs, dim=None, keep_dim=False):
    if dim is None:
        return torch.sum(inputs)
    output = torch.sum(inputs, dim)
    if not keep_dim:
        return output
    else:
        return expand_dims(output, dim)
        
    
def pairwise_add(u, v=None, is_batch=False):
    """
    performs a pairwise summation between vectors (possibly the same)
    can also be performed on batch of vectors.
    Parameters:
    ----------
    u, v: Tensor (m,) or (b,m)

    Returns: 
    ---------
    Tensor (m, n) or (b, m, n)
    
    """
    u_shape = u.size()
    if v is None:
        v = u
    v_shape = v.size()

    if len(u_shape) > 2 and not is_batch:
        raise ValueError("Expected at most 2D tensor or 3D tensor with batch")
    if len(v_shape) > 2 and not is_batch:
        raise ValueError("Expected at most 2D tensor or 3D tensor with batch")

    m = u_shape[0] if not is_batch else u_shape[1]
    n = v_shape[0] if not is_batch else v_shape[1]
    
    u = expand_dims(u, axis=-1)
    new_u_shape = list(u.size())
    new_u_shape[-1] = n
    U_ = u.expand(*new_u_shape)

    v = expand_dims(v, axis=-2)
    new_v_shape = list(v.size())
    new_v_shape[-2] = m
    V_ = v.expand(*new_v_shape)

    return U_ + V_

def to_device(src, ref):
    return src.cuda(ref.get_device()) if ref.is_cuda else src

def cumprod(inputs, dim = 1, exclusive=True):
    # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/g3doc/api_docs/python/functions_and_classes/shard0/tf.cumprod.md

    if type(inputs) is not Variable:
        temp = torch.cumprod(inputs, dim)
        if not exclusive:
            return temp
        else:
            temp =  temp / (inputs[0].expand_as(temp) + 1e-8)
            temp[-1] = temp[-1]/(inputs[-1]+1e-8)
            return temp
    else:
        shape_ = inputs.size()
        ndim = len(shape_)
        n_slot = shape_[dim]
        slice_ = [slice(0,None,1) for _ in range(ndim)]
        results = [[]] * n_slot
            
        for ind in range(0, n_slot):   
            this_slice, last_slice = copy(slice_), copy(slice_)
            this_slice[dim] = ind
            last_slice[dim] = ind-1      
            this_slice = tuple(this_slice)
            last_slice = tuple(last_slice)
            if exclusive: 
                if ind > 0:   
                    results[ind]  = results[ind-1]*inputs[last_slice]
                else:
                    results[ind] =  torch.div(inputs[this_slice], inputs[this_slice]+1e-8)
            else:    
                if ind > 0:   
                    results[ind]  = results[ind - 1]*inputs[this_slice]
                else:
                    results[ind] =  inputs[this_slice]
        
        return torch.stack(results, dim)

            
def expand_dims(input, axis=0):
    input_shape = list(input.size())
    if axis < 0:
        axis = len(input_shape) + axis + 1
    input_shape.insert(axis, 1)
    return input.view(*input_shape)


def matmal(left, right):
    '''
    left is of size (*N, n1,n2), where N is a list
    right is of size(*M, m1,m2), where M is a list
    output is of size
    '''
    pass

def cosine_distance(memory_matrix, cos_keys):
    """
    compute the cosine similarity between keys to each of the 
    memory slot.

    Parameters:
    ----------
    memory_matrix: Tensor (batch_size, mem_slot, mem_size)
        the memory matrix to lookup in
    keys: Tensor (batch_size, mem_size, number_of_keys)
        the keys to query the memory with
    strengths: Tensor (batch_size, number_of_keys, )
        the list of strengths for each lookup key
    
    Returns: Tensor (batch_size, mem_slot, number_of_keys)
        The list of lookup weightings for each provided key
    """
    memory_norm = torch.norm(memory_matrix, 2, 2, keepdim=True)
    keys_norm = torch.norm(cos_keys, 2, 1,keepdim=True)
    
    normalized_mem  = torch.div(memory_matrix, memory_norm.expand_as(memory_matrix) + 1e-8)
    normalized_keys = torch.div(cos_keys, keys_norm.expand_as(cos_keys) + 1e-8)
    
    out =  torch.bmm(normalized_mem, normalized_keys)
    
    #print(normalized_keys)
    #print(out)
    #apply_dict(locals())
    
    return out
def softmax(input, axis=1):
    """ 
    Apply softmax on input at certain axis.
    
    Parammeters:
    ----------
    input: Tensor (N*L or rank>2)
    axis: the axis to apply softmax
    
    Returns: Tensor with softmax applied on that dimension.
    """
    
    input_size = input.size()
    
    trans_input = input.transpose(axis, len(input_size)-1)
    trans_size = trans_input.size()

    input_2d = trans_input.contiguous().view(-1, trans_size[-1])
    
    soft_max_2d = F.softmax(input_2d)
    
    soft_max_nd = soft_max_2d.view(*trans_size)
    #apply_dict(locals())
    return soft_max_nd.transpose(axis, len(input_size)-1)
