import flax
import jax
import jax.numpy as jnp
import flax.linen as nn
from einops import rearrange, repeat
from typing import (Any, Callable, Iterable, List, Optional, Sequence, Tuple,
                    Union)
import time
from tqdm import trange

from flax.linen.initializers import lecun_normal, zeros, he_normal
from jax import eval_shape
from jax import lax
import numpy as np
from xcnn import CNN, XCNN


PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any  # this could be a real type?
Array = Any
PrecisionLike = Union[None, str, lax.Precision, Tuple[str, str],
                      Tuple[lax.Precision, lax.Precision]]
PaddingLike = Union[str, int, Sequence[Union[int, Tuple[int, int]]]]
default_kernel_init = lecun_normal()


class SXConv(nn.Module):
    features: int
    kernel_size: int
    strides: int
    padding: PaddingLike = 'SAME'
    activation: str = "relu"
    norm: str = "layernorm"

    @nn.compact
    def __call__(self, x):
        # ------------------------------------------------------------------
        # Components
        # ------------------------------------------------------------------
        Conv = nn.Conv(
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
        B, *H, C = x.shape
        D = len(H)
        axes = {f"h{i}": v for i,v in enumerate(H)}
        keys = list(axes.keys())
        init = " ".join(keys)

        merge = -float("inf") * jnp.ones((B, *H, C))
        for i in range(D):
            node = keys[i]
            rest = " ".join(keys[:i] + keys[i+1:])
            n = rearrange(x, f"b {init} c -> (b {rest}) {node} c")
            n = Conv(n)
            n = rearrange(n, f"(b {rest}) {node} c -> b {init} c", **axes)
            n = Norm(n)
            n = Act(n)
            merge = jnp.maximum(merge, n)
        return merge

class SXCNN1(nn.Module):
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
        for _ in range(4):
            x = SXConv(
                features=self.hidden_dim,
                kernel_size=3,
                strides=1,
                padding="SAME",
                activation="relu",
                norm="layernorm"
            )(x)

        # ------------------------------------------------------------------
        # Global Pooling
        # ------------------------------------------------------------------
        pool_axes = tuple(range(1, D+1))
        x = jnp.max(x, axis=pool_axes)

        # ------------------------------------------------------------------
        # Head
        # ------------------------------------------------------------------
        x = nn.Dense(
            features=self.out_dim,
        )(x)

        return x

class SXCNN2(nn.Module):
    out_dim: int
    hidden_dim: int
    
    @nn.compact
    def __call__(self, x):
        # x (B, H1, H2, ..., C), Hi=16, C=1
        B, *H, C = x.shape
        D = len(H)
        hidden_dim = 2*self.hidden_dim

        # ------------------------------------------------------------------
        # Axial Layers
        # ------------------------------------------------------------------
        for _ in range(5):
            x = SXConv(
                features=hidden_dim,
                kernel_size=3,
                strides=1,
                padding="SAME",
                activation="relu",
                norm="layernorm"
            )(x)

        # ------------------------------------------------------------------
        # Global Pooling
        # ------------------------------------------------------------------
        pool_axes = tuple(range(1, D+1))
        x = jnp.max(x, axis=pool_axes)

        # ------------------------------------------------------------------
        # Head
        # ------------------------------------------------------------------
        x = nn.Dense(
            features=self.out_dim,
        )(x)

        return x


def main():
    rng = jax.random.PRNGKey(0)
    x = jnp.ones((1, 16, 16, 16, 1))

    model = SXCNN1(out_dim=1, hidden_dim=128)
    params = model.init(rng, x)
    total_params = sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(params))
    print("SXCNN1 #Params", total_params)
    output = model.apply(params, x)
    print(output.shape)  # Should print (1, 1)

    model = SXCNN2(out_dim=1, hidden_dim=128)
    params = model.init(rng, x)
    total_params = sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(params))
    print("SXCNN2 #Params", total_params)
    output = model.apply(params, x)
    print(output.shape)  # Should print (1, 1)

    model = CNN(out_dim=1, hidden_dim=128)
    params = model.init(rng, x)
    total_params = sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(params))
    print("CNN 3D #Params", total_params)
    output = model.apply(params, x)
    print(output.shape)  # Should print (1, 1)

def wallclock():
    rng = jax.random.PRNGKey(0)
    x = jnp.ones((1, 16, 16, 16, 1))

    model1 = SXCNN1(out_dim=1, hidden_dim=128)
    params1 = model1.init(rng, x) 
    model2 = SXCNN2(out_dim=1, hidden_dim=128)
    params2 = model2.init(rng, x)
    model3 = XCNN(out_dim=1, hidden_dim=128)
    params3 = model3.init(rng, x)

    # forward1 = jax.jit(lambda x: model1.apply(params1, x))
    # forward2 = jax.jit(lambda x: model2.apply(params2, x))
    # forward3 = jax.jit(lambda x: model3.apply(params3, x))
    forward1 = lambda x: model1.apply(params1, x)
    forward2 = lambda x: model2.apply(params2, x)
    forward3 = lambda x: model3.apply(params3, x)
    _ = forward1(x)
    _ = forward2(x)
    _ = forward3(x)

    time1 = 0
    time2 = 0
    time3 = 0
    iters = 100
    for i in trange(iters):
        begin2 = time.time()
        _ = forward2(x)
        end2 = time.time()
        begin1 = time.time()
        _ = forward1(x)
        end1 = time.time()
        begin3 = time.time()
        _ = forward3(x)
        end3 = time.time()
        if i !=0:
            time1 += (end1 - begin1) / (iters-1)
            time2 += (end2 - begin2) / (iters-1)
            time3 += (end3 - begin3) / (iters-1)
    print("SXCNN", time1)
    print("SXCNN-L", time2)
    print("GXCNN", time3)


if __name__=="__main__":
    # main()
    wallclock()
