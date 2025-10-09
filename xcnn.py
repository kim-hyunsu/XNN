import flax
import jax
import jax.numpy as jnp
import flax.linen as nn
from einops import rearrange, repeat
from typing import (Any, Callable, Iterable, List, Optional, Sequence, Tuple,
                    Union)
import time

from flax.linen.initializers import lecun_normal, zeros, he_normal
from jax import eval_shape
from jax import lax
import numpy as np


PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any  # this could be a real type?
Array = Any
PrecisionLike = Union[None, str, lax.Precision, Tuple[str, str],
                      Tuple[lax.Precision, lax.Precision]]
PaddingLike = Union[str, int, Sequence[Union[int, Tuple[int, int]]]]
default_kernel_init = lecun_normal()

class XLift(nn.Module):
    features: int
    kernel_size: int
    strides: int
    padding: str = "SAME"
    activation: str = "relu"
    norm: str = "layernorm"

    @nn.compact
    def __call__(self, x):
        B, *H, C = x.shape
        D = len(H)
        # ------------------------------------------------------------------
        # Components
        # ------------------------------------------------------------------
        Conv = nn.Conv( # 2D conv
            features=self.features,
            kernel_size=(self.kernel_size, self.kernel_size),
            strides=(self.strides, self.strides),
            padding=self.padding,
            kernel_init=he_normal()
        )
        if self.activation == "relu":
            Act = nn.relu 
        elif self.activation == "gelu":
            Act = nn.gelu
        else:
            raise NotImplementedError(f"Activation {self.activation} not implemented")

        if self.norm == "layernorm":
            Norm = nn.LayerNorm()
        else:
            raise NotImplementedError(f"Norm {self.norm} not implemented")

        # ------------------------------------------------------------------
        # Operations
        # ------------------------------------------------------------------
        x_list = [jnp.moveaxis(x, i+1, -2) for i in range(D)]

        new_x_list = []
        for x in x_list:
            # x (B, H1, H2, ..., C), Hi=16, C=1
            B, *H, C = x.shape
            D = len(H)
            axes = {f"h{i}": v for i,v in enumerate(H)}
            keys = list(axes.keys())
            init = " ".join(keys)
            K = self.kernel_size
            S = self.strides
            out_size = [H[i] for i in range(D)] # TODO consider padding

            node = keys[-1]
            nhbr_list = keys[:-1]
            merge = -float("inf") * jnp.ones((B, *out_size, C))
            for j in range(D-1):
                nhbr = nhbr_list[j]
                rest_list = nhbr_list[:j] + nhbr_list[j+1:]
                rest = " ".join(rest_list)
                rest_dict = {k:axes[k] for k in rest_list}
                n = rearrange(x, f"b {init} c -> (b {rest}) {nhbr} {node} c")
                n = Conv(n)
                n = rearrange(n, f"(b {rest}) {nhbr} {node} c -> b {init} c", **rest_dict)
                n = Norm(n)
                n = Act(n)
                merge = jnp.maximum(merge, n)
            new_x_list.append(merge)

        return new_x_list

class XConv(nn.Module):
    features: int
    kernel_size: int
    strides: int
    padding: PaddingLike = 'SAME'
    activation: str = "relu"
    norm: str = "layernorm"

    @nn.compact
    def __call__(self, x_list):
        # ------------------------------------------------------------------
        # Components
        # ------------------------------------------------------------------
        NodeConv = nn.Conv(
            features=self.features,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            kernel_init=he_normal() 
        )
        NhbrConv = nn.Conv(
            features=self.features,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            kernel_init=he_normal() 
        )
        if self.activation == "relu":
            Act = nn.relu 
        elif self.activation == "gelu":
            Act = nn.gelu
        else:
            raise NotImplementedError(f"Activation {self.activation} not implemented")

        if self.norm == "layernorm":
            Norm = nn.LayerNorm()
        else:
            raise NotImplementedError(f"Norm {self.norm} not implemented")

        # ------------------------------------------------------------------
        # Operation
        # ------------------------------------------------------------------
        new_x_list = []
        for i, x in enumerate(x_list):
            B, *H, C = x.shape
            D = len(H)
            axes = {f"h{i}": v for i,v in enumerate(H)}
            keys = list(axes.keys())
            init = " ".join(keys)
            K = self.kernel_size
            S = self.strides
            out_size = [int((H[i]-K)/S+1) for i in range(D)]

            node = keys[-1]
            nhbr = keys[-2]
            rest = " ".join(keys[:-2])
            rest_dict = {f"h{i}": v for i,v in enumerate(H[:-2])}
            n = rearrange(x, f"b {init} c -> (b {rest}) {nhbr} {node} c")
            n = NodeConv(n)
            n = rearrange(n, f"(b {rest}) {nhbr} {node} c -> b {init} c", **rest_dict)
            n = Norm(n)
            n = Act(n)
            for j, y in enumerate(x_list):
                if i == j:
                    continue
                b, *h, c = y.shape
                out_size = [int((h[i]-K)/S+1) for i in range(D)]
                rest_dict = {f"h{k}": v for k,v in enumerate(h[:-2])}
                m = rearrange(y, f"b {init} c -> (b {rest}) {nhbr} {node} c")
                m = NhbrConv(m)
                m = rearrange(m, f"(b {rest}) {nhbr} {node} c -> b {init} c", **rest_dict)
                m = Norm(m)
                m = Act(m)
                m = jnp.moveaxis(m, -2, j+1)
                m = jnp.moveaxis(m, i+1, -2)
                n = jnp.maximum(n, m)
            new_x_list.append(n)
        return new_x_list

class XCNN(nn.Module):
    out_dim: int
    hidden_dim: int
    
    @nn.compact
    def __call__(self, x):
        # x (B, H1, H2, ..., C), Hi=16, C=1
        B, *H, C = x.shape
        D = len(H)

        # ------------------------------------------------------------------
        # Axial Layers
        # ------------------------------------------------------------------
        x_list = XLift(
            features=self.hidden_dim,
            kernel_size=3,
            strides=1,
            padding="SAME",
            activation="relu",
            norm="layernorm"
        )(x)

        x_list = XConv(
            features=self.hidden_dim,
            kernel_size=3,
            strides=1,
            padding="SAME",
            activation="relu",
            norm="layernorm"
        )(x_list)

        x_list = XConv(
            features=self.hidden_dim,
            kernel_size=3,
            strides=1,
            padding="SAME",
            activation="relu",
            norm="layernorm"
        )(x_list)

        # ------------------------------------------------------------------
        # Axial Pooling
        # ------------------------------------------------------------------
        x_list = [jnp.moveaxis(x, -2, i+1) for i,x in enumerate(x_list)]
        x_list = jnp.stack(x_list, axis=0)
        x = jnp.max(x_list, axis=0)

        # ------------------------------------------------------------------
        # Global Pooling
        # ------------------------------------------------------------------
        # pool_axes = tuple(range(1, D+1))
        # x = jnp.max(x, axis=pool_axes)

        # # ------------------------------------------------------------------
        # # Head
        # # ------------------------------------------------------------------
        # x = nn.Dense(
        #     features=self.out_dim,
        # )(x)

        return x


class CNN(nn.Module):
    out_dim: int
    hidden_dim: int
    
    @nn.compact
    def __call__(self, x):
        # x (B, H1, H2, H3, C), C=1
        B, H, W, D, C = x.shape
        D = 3

        # ------------------------------------------------------------------
        # Axial Layers
        # ------------------------------------------------------------------
        x = nn.Conv(
            features=self.hidden_dim,
            kernel_size=(3,3,3),
            strides=(1,1,1),
            padding="SAME",
        )(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)

        x = nn.Conv(
            features=self.hidden_dim,
            kernel_size=(3,3,3),
            strides=(1,1,1),
            padding="SAME",
        )(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)

        x = nn.Conv(
            features=self.hidden_dim,
            kernel_size=(3,3,3),
            strides=(1,1,1),
            padding="SAME",
        )(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)

        # ------------------------------------------------------------------
        # Global Pooling
        # ------------------------------------------------------------------
        # x = jnp.max(x, axis=(1,2,3))

        # # ------------------------------------------------------------------
        # # Head
        # # ------------------------------------------------------------------
        # x = nn.Dense(
        #     features=self.out_dim,
        # )(x)

        return x



if __name__ == "__main__":
    # --------------------------------------------------------------------
    # XCNN
    # --------------------------------------------------------------------
    rng = jax.random.PRNGKey(0)
    x = jax.random.normal(rng, (2, 16, 16, 16, 1))  # (B, H1, H2, C)
    px = jnp.moveaxis(x, -3, 1)
    model = XCNN(out_dim=10, hidden_dim=32)
    # model = CNN(out_dim=10, hidden_dim=32)
    params = model.init(rng, x)
    begin = time.time()
    y = model.apply(params, x)
    y_px = model.apply(params, px)
    print("y_px", y_px) 
    print("XCNN", y.shape, "time", time.time()-begin)  # should be (2, 10)
    py = jnp.moveaxis(y, -3, 1)
    print("diff", jnp.sum((py-y_px)**2))

    # --------------------------------------------------------------------
    # 3D CNN
    # --------------------------------------------------------------------
    # model2 = CNN(out_dim=10, hidden_dim=32)
    # x1 = jnp.ones((2, 16, 16, 1))
    # x2 = jnp.ones((2, 16, 16, 16, 16, 1))

    # for x in [x1,x2]:
    #     ndim = x.ndim
    #     if ndim < 5:
    #         ex = 5 - ndim
    #         x = jnp.expand_dims(x, tuple(range(1, ex+1)))
    #         x = jnp.pad(x, [(0,0)]+[(0, 15)]*ex+[(0,0)]*(ndim-2)+[(0,0)])
    #     else:
    #         x = x.reshape(*x.shape[0:3], -1, x.shape[-1])
    #     params = model2.init(rng, x)
    #     begin = time.time()
    #     y = model2.apply(params, x)
    #     print("CNN", y.shape, "time", time.time()-begin)  # should be (2, 10)
