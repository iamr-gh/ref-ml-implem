from jaxtyping import Float
import numpy as np
from torch import Tensor
from torch import torch

# let's look at implementing an attention mechanism

# Q, K, V projections into a hidden dim
# y = softmax(QK/sqrt(d))/V


# one attn no batch
def single_attn(
    x: Float[Tensor, "seq_len emb_dim"],
    W_q: Float[Tensor, "emb_dim d"],
    W_k: Float[Tensor, "emb_dim d"],
    W_v: Float[Tensor, "emb_dim d_v"],
) -> Float[Tensor, "seq_len d_v"]:

    d = W_q.shape[-1]

    Q = x @ W_q
    K = x @ W_k
    V = x @ W_v

    S = (Q @ K.T) / np.sqrt(d)
    A = S.softmax(-1)

    return A @ V


def main():
    x = torch.zeros(2, 5)
    w_q = torch.zeros(5, 3)
    w_k = torch.zeros(5, 3)
    w_v = torch.zeros(5, 4)

    y = single_attn(x, w_q, w_k, w_v)
    print(y.shape)

    print("Hello from jax!")


if __name__ == "__main__":
    main()
