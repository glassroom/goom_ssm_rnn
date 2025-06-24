# coding: utf-8

import math
import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

import generalized_orders_of_magnitude as goom
import torch_parallel_scan as tps


class LogMultiDenseExp(nn.Module):
    """
    Given log_A1 (d x d), log_B1 (n x d), and log_A2 (d x d), log_B2 (n x d),
    compute log(A1 @ A2) and log(B1 @ A2 + B2), over generalized orders of
    magnitude (GOOMs), broadcasting over any preceding dimensions.

    Args:
        n: number of vectors per matrix.
        d: number of features per vector.

    Inputs:
        log_A1_atop_B1: complex tensor of shape [..., d + n, d], with each
            log_A1 (d x d) stacked atop its corresponding log_B1 (n x d).
        log_A2_atop_B2: complex tensor of shape [..., d + n, d], with each
            log_A2 (d x d) stacked atop its corresponding log_B2 (n x d).

    Output:
        updated_log_A2_atop_B2: complex tensor of shape [..., d + n, d], with
            each log(A1 @ A2) (d x d) stacked atop log(B1 @ A2 + B2) (n x d).
    """
    def __init__(self, n, d):
        super().__init__()
        self.register_buffer('log_0s_atop_I', goom.log(
            torch.cat([
                torch.zeros(d, n),
                torch.eye(n, n)
            ], dim=0)))                                           # [d + n, n]

    def forward(self, log_A1_atop_B1, log_A2_atop_B2):
        _szs = log_A1_atop_B1.shape[:-2]
        log_0s_atop_I = self.log_0s_atop_I.expand(*_szs, -1, -1)  # [..., d + n, n]

        log_A1_atop_B1_with_0s_atop_I = torch.cat([
            log_A1_atop_B1,
            log_0s_atop_I,
        ], dim=-1)                                                # [..., d + n, d + n]

        updated_log_A2_atop_B2 = goom.log_matmul_exp(
            log_A1_atop_B1_with_0s_atop_I,
            log_A2_atop_B2,
        )                                                         # [..., d + n, d]
        return updated_log_A2_atop_B2


class SSMoverGOOMs(nn.Module):
    """
    Computes the linear-time-invariant non-diagonal state-space system

        x_t = A x_{t-1} + B u_t
        y_t = C x_t + D u_t,

    where u_t, x_t, and y_t are input, hidden, and output state vectors,
    and A, B, C, and D are matrix parameters, via a parallel prefix scan,
    over generalized orders of magnitude (GOOMs), allowing real values to
    fluctuate over a greater dynamic range than previously possible as
    proposed in "Generalized Orders of Magnitude for Scalable, Parallel,
    High-Dynamic-Range Computation" (Heinsen and Kozachkov, 2025)

    Args:
        d_inp: number of input features per step
        d_out: number of output features per step
        n_hid: number of hidden state vectors per step
        d_hid: number of elements per hidden state vector

    Inputs:
        u: tensor of shape [..., seq_len, d_inp]

    Output:
        y: tensor of shape [..., seq_len, d_out]
    """
    def __init__(self, d_inp, d_out, n_hid, d_hid):
        super().__init__()
        self.d_inp, self.d_out, self.n_hid, self.d_hid = (d_inp, d_out, n_hid, d_hid)
        self.register_buffer('log_I', goom.log(torch.eye(n_hid)))
        self.init_states = nn.Parameter(torch.empty(n_hid, d_hid).uniform_(-0.1, 0.1))
        self.A = nn.Parameter(nn.init.orthogonal_(torch.empty(d_hid, d_hid), gain=0.99))
        self.B = nn.Parameter(torch.empty(d_inp, n_hid * d_hid).uniform_(-1, 1) / math.sqrt(d_inp))
        self.C = nn.Parameter(torch.empty(n_hid * d_hid, d_out).uniform_(-1, 1) / math.sqrt(n_hid * d_hid))
        self.D = nn.Parameter(torch.empty(d_inp, d_out).uniform_(-1, 1) / math.sqrt(d_inp))
        self.lmde = LogMultiDenseExp(n_hid, d_hid)

    def forward(self, u, continue_prev=False):
        # Prepare sequential weights A and biases B u_t:
        A = self.A.expand(*u.shape[:-1], -1, -1)                                  # [..., seq_len, d_hid, d_hid]
        Bu = torch.matmul(u, self.B).view(*u.shape[:-1], self.n_hid, self.d_hid)  # [..., seq_len, n_hid, d_hid]

        # Compute x_t in parallel over generalized orders of magnitude (GOOMs):
        log_A_atop_Bu = goom.log(torch.cat([A, Bu], dim=-2))                      # [..., seq_len, d_hid + n_hid, d_hid]
        log_cum_A_atop_Bu = tps.prefix_scan(log_A_atop_Bu, self.lmde, dim=-3)     # [..., seq_len, d_hid + n_hid, d_hid]

        if continue_prev:
            log_x0 = self.log_prev_states                                         # [..., 1, n_hid, d_hid] 
        else:
            log_x0 = goom.log(self.init_states)                                   # [n_hid, d_hid]

        log_I = self.log_I.expand(*log_x0.shape[:-2], -1, -1)                     # [..., 1, n_hid, n_hid], or [n_hid, n_hid]
        log_x0_with_I = torch.cat([log_x0, log_I], dim=-1)                        # [..., 1, n_hid, d_hid + n_hid], or [n_hid, d_hid + n_hid]

        log_x = goom.log_matmul_exp(log_x0_with_I, log_cum_A_atop_Bu)             # [..., seq_len, n_hid, d_hid]
        self.log_prev_states = log_x[..., -1:, :, :,].detach()                    # [..., 1, n_hid, d_hid]

        # Log-scale GOOMs before exponentiation, since magnitudes may not
        # be representable as floats; then, exponentiate them to floats:
        c = log_x.real.detach().max(dim=-1, keepdim=True).values                  # [..., seq_len, n_hid, 1], log-scaling constants
        x = goom.exp(log_x - c + 2).flatten(-2)                                   # [..., seq_len, n_hid * d_hid], between -exp(2) and exp(2)

        # Compute output states y_t:
        Cx = torch.matmul(x, self.C)                                              # [..., seq_len, d_out]
        Du = torch.matmul(u, self.D)                                              # [..., seq_len, d_out]
        y = Cx + Du                                                               # [..., seq_len, d_out]

        return y


class ResidualRecurrentLayer(nn.Module):    
    """
    Update each token with information from its preceding tokens, in parallel,
    capturing sequential dependencies with the module `SSMoverGOOMs`.

    Args:
        d_emb: number of embedding features per step
        n_hid: number of hidden state vectors per step
        d_hid: number of elements per hidden state vector

    Inputs:
        inp: tensor of shape [..., seq_len, d_emb]

    Output:
        out: tensor of shape [..., seq_len, d_emb]
    """
    def __init__(self, d_emb, n_hid, d_hid):
        super().__init__()
        self.d_emb, self.n_hid, self.d_hid = (d_emb, n_hid, d_hid)
        self.lnorm = nn.LayerNorm(d_emb)
        self.ssm_over_gooms = SSMoverGOOMs(d_emb, d_emb * 2, n_hid, d_hid)
        self.feedfwd = nn.Sequential(nn.GLU(dim=-1), nn.Linear(d_emb, d_emb, bias=False))

    def forward(self, inp, continue_prev=False):
        x = self.lnorm(inp)                        # [..., seq_len, d_emb]
        x = self.ssm_over_gooms(x, continue_prev)  # [..., seq_len, d_emb * 2]
        res = self.feedfwd(x)                      # [..., seq_len, d_emb]
        return inp + res


class GenerativeRNN(nn.Module):
    """
    Given a sequence of token ids, generate the next token id.

    Args:
        vocab_sz: int, number of token ids in vocab.
        d_emb: int, number of embedding features per token
        n_res: int, number of residual recurrent layers
        n_hid: int, number of hidden state vectors per token
        d_hid: int, number of features per hidden state vector

    Input shape: token_ids: [..., n_toks], sequence of token ids.
    Output shape: predicted logits [..., n_toks, vocab_sz].
    """
    def __init__(self, vocab_sz, d_emb, n_res, n_hid, d_hid):
        super().__init__()
        self.vocab_sz, self.d_emb, self.n_res, self.n_hid, self.d_hid = (vocab_sz, d_emb, n_res, n_hid, d_hid)
        self.embed = nn.Embedding(vocab_sz, d_emb)
        self.res_layers = nn.Sequential(*[ResidualRecurrentLayer(d_emb, n_hid, d_hid) for _ in range(n_res)])
        self.lnorm = nn.LayerNorm(d_emb)
        self.head = nn.Linear(d_emb, vocab_sz, bias=False)  # conventional language generation head
        self.embed.weight = self.head.weight                # tie weights as one, as is conventional
        
    def body(self, token_ids, continue_prev=False):
        x = self.embed(token_ids)
        for res_layer in self.res_layers:
            x = res_layer(x, continue_prev)
        x = self.lnorm(x)
        return x

    def forward(self, token_ids, continue_prev=False):
        x = self.body(token_ids, continue_prev)
        x = self.head(x)
        return x

    # Convenience methods:

    def get_param_groups(self, weight_decay):
        "Given a weight decay, returns two parameter groups for training."
        decay_attrs = { nn.Embedding: ['weight'], nn.Linear: ['weight'], SSMoverGOOMs: ['init_states', 'A', 'B', 'C', 'D'], }
        decay_modules = set(m for m in self.modules() if type(m) in decay_attrs.keys())
        decay_ids = set(id(getattr(m, attr)) for m in decay_modules for attr in decay_attrs[type(m)])
        return [
            { 'params': [p for p in self.parameters() if id(p) in decay_ids], 'weight_decay': weight_decay, },
            { 'params': [p for p in self.parameters() if id(p) not in decay_ids], 'weight_decay': 0.0, },
        ]

    def compute_loss_and_metrics(self, preds, targets):
        "Compute and return model loss and a dict of custom metrics."
        preds = preds.view(-1, preds.size(-1))
        targets = targets.flatten()
        loss = F.cross_entropy(preds, targets)

        n_targets = targets.numel()
        n_correct = (preds.argmax(dim=-1) == targets).long().sum().item()
        metrics = { 'accuracy': n_correct / n_targets, }

        return loss, metrics

    @torch.no_grad()
    def generate(self, token_ids, n_new, temp=1.0, topk=None, continue_prev=False, show_progress=False):
        "Given a sequence of token ids, generate new token ids."
        assert self.training is False, "Model should be in eval mode."
        generated_ids = []
        cp_states = [continue_prev] + [True] * (n_new - 1)
        for cp_state in (tqdm(cp_states) if show_progress else cp_states):
            hidden_states = self.body(token_ids, continue_prev=cp_state)
            logits = self.head(hidden_states[..., -1, :]) / temp
            if topk is not None:
                min_of_topk = logits.topk(topk, dim=-1).values.min(dim=-1, keepdim=True).values
                logits[logits < min_of_topk] = float('-inf')
            token_ids = torch.multinomial(logits.softmax(dim=-1), num_samples=1) 
            generated_ids.append(token_ids)
        return torch.cat(generated_ids, dim=-1)
