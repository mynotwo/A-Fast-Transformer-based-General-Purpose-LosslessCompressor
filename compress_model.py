# coding=utf-8

import numpy as np
import torch
import time
import numerator_and_denominator as num_and_den


def valid_feature_type(feature_type):
  bool1 = feature_type in ['relu', 'elu+1', 'sqr', 'favor+']
  bool2 = feature_type.startswith('favor+') and feature_type.split(
      '_')[1].isdigit()
  return bool1 or bool2


class SLiMPerformer(torch.nn.Module):

  def __init__(self, vocab_size, vocab_dim,  hidden_dim, n_layers, ffn_dim, n_heads, feature_type, compute_type):
    super(SLiMPerformer, self).__init__()

    self._vocab_size = vocab_size
    self._vocab_dim = vocab_dim
    self._hidden_dim = hidden_dim
    self._scale = hidden_dim // vocab_dim
    self.input_map = torch.nn.Embedding(vocab_size, vocab_dim // 2)
    self.output_logit_map = torch.nn.Linear(hidden_dim, vocab_size)

    self.layers = torch.nn.ModuleList([
        SLiMPerformerLayer(hidden_dim, ffn_dim, n_heads, feature_type,
                           compute_type) for _ in range(n_layers)
    ])

  def forward(self, x):

    x = self.input_map(x)
    x = self._concat_pos_embs(x, 0)
    bs, seqlen, vlen = x.shape
    
    x = x.reshape(bs, seqlen // self._scale, vlen*self._scale)
    for layer in self.layers:
      x = layer.full_forward(x, layer.attention.sample_rfs(x.device))
    
    x = self.output_logit_map(x)

    return x

  def full_loss(self,
                inputs,
                with_grad=True):

    logits = self.forward(inputs[:, :-1])
    logits = logits.transpose(1, 2)
    loss = torch.nn.functional.cross_entropy(
            logits[:, :, -1], inputs[:, -1], reduction='mean')

    if with_grad:
      loss.backward()

    return loss, logits

  def _concat_pos_embs(self, x, start_index):

    pos_emb_size = self._vocab_dim // 2

    positions = torch.arange(
        start_index, start_index + x.shape[1], dtype=x.dtype, device=x.device)
    freqs = torch.exp(
        torch.arange(0, pos_emb_size, 2, dtype=x.dtype, device=x.device) *
        (-np.log(10000) / pos_emb_size))
    args = positions[None, :, None] * freqs[None, None, :]
    sin_pos_embs = torch.sin(args) * torch.ones_like(x[:, :1, :1])
    cos_pos_embs = torch.cos(args) * torch.ones_like(x[:, :1, :1])
    return torch.cat([x, sin_pos_embs, cos_pos_embs], 2)


class SLiMPerformerLayer(torch.nn.Module):

  def __init__(self, hidden_dim, ffn_dim, n_heads, feature_type, compute_type):

    super(SLiMPerformerLayer, self).__init__()

    self.attention = MultiHeadAttention(feature_type, n_heads, hidden_dim,
                                        compute_type)

    self.U_map = torch.nn.Linear(hidden_dim, ffn_dim)
    self.V_map = torch.nn.Linear(ffn_dim, hidden_dim)
    self.layernorm1 = torch.nn.LayerNorm(hidden_dim)
    self.layernorm2 = torch.nn.LayerNorm(hidden_dim)

  def full_forward(self, x, rfs):

    skip = x

    x = self.layernorm1(x)

    x = self.attention.full_forward(x, rfs)

    x = skip + x
    
    x = self._ffn(x)
    x = self._ffn(x)

    return x

  def _ffn(self, x):

    skip = x

    x = self.layernorm2(x)

    x = self.U_map(x)
    x = torch.nn.functional.gelu(x)
    x = self.V_map(x)

    x = skip + x

    return x


class MultiHeadAttention(torch.nn.Module):
  """Explicit multihead attention using prefix sum."""

  def __init__(self, feature_type, n_heads, hidden_dim, compute_type):

    super(MultiHeadAttention, self).__init__()

    self._feature_type = feature_type
    self._n_heads = n_heads
    self._hidden_dim = hidden_dim
    self._compute_type = compute_type

    self.q_map = torch.nn.Linear(hidden_dim, hidden_dim)
    self.k_map = torch.nn.Linear(hidden_dim, hidden_dim)
    self.v_map = torch.nn.Linear(hidden_dim, hidden_dim)

  def full_forward(self, x, rfs):

    queries, keys, values = self._get_queries_keys_values(x, rfs)

    num_sums, den_sums = self.init_sums(x.device)

    if self._compute_type == 'iter':
      num, _ = num_and_den.num_iter(queries, keys, values, num_sums)
      den, _ = num_and_den.den_iter(queries, keys, den_sums)
    elif self._compute_type == 'ps':
      num, _ = num_and_den.num_ps(queries, keys, values, num_sums, False)
      den, _ = num_and_den.den_ps(queries, keys, den_sums, False)
    else:
      num, _ = num_and_den.num_ps(queries, keys, values, num_sums, True)
      den, _ = num_and_den.den_ps(queries, keys, den_sums, True)

    num = torch.transpose(num, 0, 1)
    den = torch.transpose(den, 0, 1)

    outputs = num / (den[Ellipsis, None] + 1e-16)
    outputs = outputs.reshape(x.shape)

    return outputs

  def init_sums(self, device):

    head_dim = self._hidden_dim // self._n_heads

    if self._feature_type.startswith('favor+_'):
      splitted = self._feature_type.split('_')
      feature_dim = int(splitted[1]) * head_dim
    else:
      feature_dim = head_dim

    num_sums = torch.zeros([1, self._n_heads, feature_dim, head_dim],
                           device=device)
    den_sums = torch.zeros([1, self._n_heads, feature_dim], device=device)

    return num_sums, den_sums


  def _get_queries_keys_values(self, inputs, rfs):

    queries = self.q_map(inputs)
    keys = self.k_map(inputs)
    values = self.v_map(inputs)

    queries = queries.reshape(
        [queries.shape[0], queries.shape[1], self._n_heads, -1])
    keys = keys.reshape([keys.shape[0], keys.shape[1], self._n_heads, -1])
    values = values.reshape(
        [values.shape[0], values.shape[1], self._n_heads, -1])

    if self._feature_type == 'relu':
      queries = torch.nn.functional.relu(queries)
      keys = torch.nn.functional.relu(keys)
    elif self._feature_type == 'elu+1':
      queries = torch.nn.functional.elu(queries) + 1
      keys = torch.nn.functional.elu(keys) + 1
    elif self._feature_type == 'sqr':
      queries = queries**2
      keys = keys**2
    elif self._feature_type == 'abs':
      queries = torch.abs(queries)
      keys = torch.abs(keys)
    else:

      head_dim = self._hidden_dim // self._n_heads

      queries = queries * np.power(head_dim, -0.25)
      queries = torch.einsum('ijkl,klm->ijkm', queries, rfs) - (queries**2).sum(
          3, keepdim=True) / 2
      queries = torch.exp(queries)

      keys = keys * np.power(head_dim, -0.25)
      keys = torch.einsum('ijkl,klm->ijkm', keys, rfs) - (keys**2).sum(
          3, keepdim=True) / 2
      keys = torch.exp(keys)

    queries = queries.transpose(0, 1)
    keys = keys.transpose(0, 1)
    values = values.transpose(0, 1)

    return queries, keys, values

  def sample_rfs(self, device):

    if not self._feature_type.startswith('favor+'):
      return None

    if self._feature_type == 'favor+':
      factor = 1
    else:
      splitted = self._feature_type.split('_')
      factor = int(splitted[1])

    head_dim = self._hidden_dim // self._n_heads

    rfs = [[
        _sample_orth_matrix(head_dim, device)[None, Ellipsis] for _ in range(factor)
    ] for _ in range(self._n_heads)]
    rfs = [torch.cat(x, 2) for x in rfs]
    rfs = torch.cat(rfs, 0)
    rfs = rfs * np.sqrt(head_dim)

    return rfs


def _sample_orth_matrix(size, device):
  """Samples orthogonal matrix to reduce variance for random features."""
  subspace = torch.randn(size, size, device=device)
  subspace = torch.tril(subspace)
  subspace = subspace / torch.sqrt((subspace**2).sum(0, keepdim=True))

  S = torch.triu(subspace.T.mm(subspace)) - 0.5 * torch.eye(
      subspace.shape[1], device=device)

  result = torch.eye(
      subspace.shape[0], device=device) - subspace.mm(torch.inverse(S)).mm(
          subspace.T)

  return result
