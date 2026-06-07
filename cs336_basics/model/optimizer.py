from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math
from torch import Tensor
from torch import nn

class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params, 
        lr: float | Tensor = 1e-3,
        betas: tuple[float | Tensor, float | Tensor] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        
        params = list(params)

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["t"] = 1
                state["m"] = torch.zeros_like(p)
                state["v"] = torch.zeros_like(p)
        
        
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
        for p in group["params"]:
            if p.grad is None:
                continue
            state = self.state[p] # Get state associated with p.
            t = state.get("t", 1) # Get iteration number from the state, or 0.
            grad = p.grad.data # Get the gradient of loss with respect to p.
            

            m = state.get("m")
            v = state.get("v")

            lr_t = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)
            p.data = p.data - lr * weight_decay * p.data
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * grad**2
            p.data = p.data - lr_t * m / (v.sqrt() + eps)
            
            state["m"] = m
            state["v"] = v
            state["t"] = t + 1 # Increment iteration number.
        return loss