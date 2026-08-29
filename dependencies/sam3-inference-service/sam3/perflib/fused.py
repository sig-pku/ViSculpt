# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

import torch
import torch.nn.functional as F

addmm_act_op = torch.ops.aten._addmm_activation


def addmm_act(activation, linear, mat1):
    if torch.is_grad_enabled():
        raise ValueError("Expected grad to be disabled.")

    if activation in [F.relu, torch.nn.ReLU]:
        activation_fn = F.relu
    elif activation in [F.gelu, torch.nn.GELU]:
        activation_fn = F.gelu
    else:
        raise ValueError(f"Unexpected activation {activation}")

    # The private fused operator is a CUDA bfloat16 optimization. Standard
    # PyTorch layers dispatch correctly on MPS/CPU and on pre-Ampere CUDA GPUs.
    use_fused_cuda = False
    if mat1.device.type == "cuda" and isinstance(linear, torch.nn.Linear):
        with torch.cuda.device(mat1.device):
            use_fused_cuda = torch.cuda.is_bf16_supported()
    if not use_fused_cuda:
        return activation_fn(linear(mat1))

    self = linear.bias.detach()
    mat2 = linear.weight.detach()
    self = self.to(torch.bfloat16)
    mat1 = mat1.to(torch.bfloat16)
    mat2 = mat2.to(torch.bfloat16)
    mat1_flat = mat1.view(-1, mat1.shape[-1])
    if activation in [F.relu, torch.nn.ReLU]:
        y = addmm_act_op(self, mat1_flat, mat2.t(), beta=1, alpha=1, use_gelu=False)
        return y.view(mat1.shape[:-1] + (y.shape[-1],))
    if activation in [F.gelu, torch.nn.GELU]:
        y = addmm_act_op(self, mat1_flat, mat2.t(), beta=1, alpha=1, use_gelu=True)
        return y.view(mat1.shape[:-1] + (y.shape[-1],))
