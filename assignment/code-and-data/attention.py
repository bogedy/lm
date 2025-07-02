from typing import Optional
from torch import nn
import torch
import torch.nn.functional as F
import math


def create_kqv_matrix(input_vector_dim, output_dim):
    return nn.Linear(input_vector_dim, output_dim)

def kqv(x, linear):
    B, N, D = x.size()
    # compute k, q, and v
    # (can do it in 1 or 2 lines.)
    kqv = linear(x)
    k, q, v = torch.split(kqv, kqv.shape[-1] // 3, dim=-1)
    return k, q, v

def attention_scores(a, b):

    B1, N1, D1 = a.size()
    B2, N2, D2 = b.size()
    assert B1 == B2
    assert D1 == D2

    # compute A (remember: we are computing *scaled* dot product attention. don't forget the scaling.
    # (can do it in 1 or 2 lines.)
    A = a @ b.transpose(-2, -1) / (D1 ** 0.5)
    return A

def create_causal_mask(embed_dim, n_heads, max_context_len):
    # Return a causal mask (a tensor) with zeroes in dimensions we want to zero out.
    # This function receives more arguments than it actually needs. This is just because
    # it is part of an assignment, and I want you to figure out on your own which arguments
    # are relevant.

    mask = torch.tril(torch.ones(max_context_len, max_context_len)).unsqueeze(0)
    return mask

def self_attention(v, A, mask = None):
    # compute sa (corresponding to y in the assignemnt text).
    # This should take very few lines of code.
    # As usual, the dimensions of v and of sa are (b x n x d).

    #### THIS IS THE FUNCTION where we do the softmax
    if mask is not None:
        seq_len = A.shape[1]
        M = mask[:, :seq_len, :seq_len]
        A = A.masked_fill(M == 0., float("-inf"))
    attention_weights = torch.softmax(A, dim=-1)  # (b x n x n)
    sa = attention_weights @ v
    # from linear algebra first principles, it makes sense to think of this like (V^T * A^T)^T 
    # Becuase I'm just picturing it from linear algebra first principles, we want the matrix V^T to ditribute over the cols of A^T
    # but this equals A @ V, so nice. 
    return sa


def self_attention_layer(x, kqv_matrix, attention_mask):
    k, q, v = kqv(x, kqv_matrix)
    att = attention_scores(k, q)
    sa = self_attention(v, att, attention_mask)
    return sa

def multi_head_attention_layer(x, kqv_matrix, n_heads, mask):
    B, N, D = x.size()
    head_dim = D // n_heads
    
    # implement multi-head attention.
    # This is most easily done using calls to self_attention_layer, each with a different
    # entry in kqv_matrices, and combining the results.

    # sa_heads = [self_attention_layer(x, kqv_matrix, mask) for kqv_matrix in kqv_matrices]
    # sa = torch.cat(sa_heads, dim=-1)

    # There is also a tricker (but more efficient) version of multi-head attention, where we do all the computation
    # using a single multiplication with a single kqv_matrix (or a single kqv_tensor) and re-arranging the results afterwards.
    # If you want a challenge, you can try and implement this. You may need to change additional places in the code accordingly.

    kqv_combined = kqv_matrix(x)  # Shape: (B, N, 3 * D)
    k, q, v = kqv_combined.chunk(3, dim=-1)
    # each with shape (B, N, D)

    # important to reshape this way so that we don't scramble our data!
    k = k.view(B, N, n_heads, head_dim).transpose(1, 2)
    q = q.view(B, N, n_heads, head_dim).transpose(1, 2)
    v = v.view(B, N, n_heads, head_dim).transpose(1, 2)

    att = (q @ k.transpose(-2, -1)) / (k.size(-1) ** 0.5)
    # shape (B, n_heads, N, N)

    if mask is not None:
        M = mask[:, :N, :N].unsqueeze(1)  # add head dimension
        att = att.masked_fill(M == 0., float("-inf"))

    att_weights = torch.softmax(att, dim=-1)
    sa = att_weights @ v  # (B, n_heads, N, head_dim)
    sa = sa.transpose(1, 2).reshape(B, N, -1)

    assert sa.size() == x.size()
    return sa


class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, n_heads, max_context_len):
        super().__init__()
        assert embed_dim % n_heads == 0
        # the linear layers used for k, q, v computations:
        # each linear is for a different head, but for all of k, q and v for this head.
        self.kqv_matrix = create_kqv_matrix(embed_dim, embed_dim * 3)
        # For use in the causal part.  "register_buffer" is used to store a tensor which is fixed but is not a parameter of the model.
        # You can then access it with: self.mask
        mask = create_causal_mask(embed_dim, n_heads, max_context_len)
        self.register_buffer("mask", mask)
        self.n_heads = n_heads
        self.embed_dim = embed_dim
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        sa = multi_head_attention_layer(x, self.kqv_matrix, self.n_heads, self.mask)
        sa = self.proj(sa)
        return sa
