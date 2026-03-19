from jaxtyping import Float
import numpy as np
import torch
from torch import Tensor

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

    # normally would get causal masked
    S = (Q @ K.T) / np.sqrt(d)

    A = S.softmax(-1)

    return A @ V


def batch_single_attn(
    x: Float[Tensor, "b seq_len emb_dim"],
    W_q: Float[Tensor, "emb_dim d"],
    W_k: Float[Tensor, "emb_dim d"],
    W_v: Float[Tensor, "emb_dim d_v"],
) -> Float[Tensor, "b seq_len d_v"]:

    d = W_q.shape[-1]

    Q = x @ W_q
    K = x @ W_k
    V = x @ W_v

    # normally would get causal masked
    S = (Q @ K.mT) / np.sqrt(d)

    A = S.softmax(-1)

    return A @ V


def batch_mha(
    x: Float[Tensor, "b seq_len emb_dim"],
    W_q: Float[Tensor, "h emb_dim d"],
    W_k: Float[Tensor, "h emb_dim d"],
    W_v: Float[Tensor, "h emb_dim d_v"],
) -> Float[Tensor, "b seq_len h_d_v"]:
    d = W_q.shape[-1]

    x_heads = x.unsqueeze(1)

    Q = x_heads @ W_q.unsqueeze(0)  # [b, h, seq_len, d]
    K = x_heads @ W_k.unsqueeze(0)  # [b, h, seq_len, d]
    V = x_heads @ W_v.unsqueeze(0)  # [b, h, seq_len, d_v]

    S = (Q @ K.mT) / np.sqrt(d)  # [b, h, seq_len, seq_len]
    A = S.softmax(-1)

    out = A @ V  # [b, h, seq_len, d_v]

    b, h, seq_len, d_v = out.shape
    out = out.transpose(1, 2)  # [b, seq_len, h, d_v]
    out = out.reshape(b, seq_len, h * d_v)

    return out


def main():
    x = torch.zeros(2, 5)
    w_q = torch.zeros(5, 3)
    w_k = torch.zeros(5, 3)
    w_v = torch.zeros(5, 4)

    y = single_attn(x, w_q, w_k, w_v)
    print(y.shape)

    x_bat = torch.zeros(4, 2, 5)
    y_bat = batch_single_attn(x_bat, w_q, w_k, w_v)

    print(y_bat.shape)

    w_q_h = torch.zeros(3, 5, 4)
    w_k_h = torch.zeros(3, 5, 4)
    w_v_h = torch.zeros(3, 5, 4)

    y_mha = batch_mha(x_bat, w_q_h, w_k_h, w_v_h)
    print(y_mha.shape)

    print("Hello from jax!")


if __name__ == "__main__":
    main()
