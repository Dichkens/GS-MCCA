import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.nn import RGCNConv, GraphConv
import numpy as np, itertools, random, copy, math
from torch.autograd import Variable

from models import MMGatedAttention

class MaskedKLDivLoss(nn.Module):
    def __init__(self):
        super(MaskedKLDivLoss, self).__init__()
        self.loss = nn.KLDivLoss(reduction='none')

    def forward(self, log_pred, target, mask):
        mask_flat = mask.view(-1)
        mask_sum = torch.sum(mask_flat)
        if mask_sum == 0:
            return torch.tensor(0.0, device=log_pred.device, dtype=log_pred.dtype)

        # Compute KL divergence loss with reduction='none' to get per-sample losses
        if log_pred.size(0) == mask_flat.size(0):
            loss_per_sample = self.loss(log_pred, target).sum(dim=1)
            masked_loss = (loss_per_sample * mask_flat) / mask_sum
            return torch.sum(masked_loss)
        else:
            if target.size(0) == log_pred.size(0):
                target_flat = target
            elif target.numel() == mask_flat.numel() * target.size(-1):
                valid = mask_flat > 0
                target_flat = target.view(-1, target.size(-1))[valid]
            else:
                raise ValueError(f"Unexpected target shape {tuple(target.size())} for mask length {mask_flat.numel()}")

            loss_per_sample = self.loss(log_pred, target_flat).sum(dim=1)
            return loss_per_sample.sum() / mask_sum

def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.actv = gelu
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        inter = self.dropout_1(self.actv(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output + x

class MaskedNLLLoss(nn.Module):
    def __init__(self, weight=None):
        super(MaskedNLLLoss, self).__init__()
        self.weight = weight
        self.loss = nn.NLLLoss(weight=weight, reduction='none')

    def forward(self, pred, target, mask):
        mask_ = mask.view(-1)
        mask_sum = torch.sum(mask_)
        if mask_sum == 0:
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        # Compute loss for all samples, then mask
        loss_per_sample = self.loss(pred, target)
        masked_loss = (loss_per_sample * mask_) / mask_sum
        return torch.sum(masked_loss)

class UnMaskedWeightedNLLLoss(nn.Module):

    def __init__(self, weight=None):
        super(UnMaskedWeightedNLLLoss, self).__init__()
        self.weight = weight
        self.loss = nn.NLLLoss(weight=weight,
                               reduction='sum')

    def forward(self, pred, target):
        """
        pred -> batch*seq_len, n_classes
        target -> batch*seq_len
        """
        if type(self.weight)==type(None):
            loss = self.loss(pred, target)
        else:
            loss = self.loss(pred, target)\
                            /torch.sum(self.weight[target])
        return loss


class SimpleAttention(nn.Module):

    def __init__(self, input_dim):
        super(SimpleAttention, self).__init__()
        self.input_dim = input_dim
        self.scalar = nn.Linear(self.input_dim,1,bias=False)

    def forward(self, M, x=None):
        """
        M -> (seq_len, batch, vector)
        x -> dummy argument for the compatibility with MatchingAttention
        """
        scale = self.scalar(M)
        alpha = F.softmax(scale, dim=0).permute(1,2,0)
        attn_pool = torch.bmm(alpha, M.transpose(0,1))[:,0,:]
        return attn_pool, alpha


class MatchingAttention(nn.Module):

    def __init__(self, mem_dim, cand_dim, alpha_dim=None, att_type='general'):
        super(MatchingAttention, self).__init__()
        assert att_type!='concat' or alpha_dim!=None
        assert att_type!='dot' or mem_dim==cand_dim
        self.mem_dim = mem_dim
        self.cand_dim = cand_dim
        self.att_type = att_type
        if att_type=='general':
            self.transform = nn.Linear(cand_dim, mem_dim, bias=False)
        if att_type=='general2':
            self.transform = nn.Linear(cand_dim, mem_dim, bias=True)
        elif att_type=='concat':
            self.transform = nn.Linear(cand_dim+mem_dim, alpha_dim, bias=False)
            self.vector_prod = nn.Linear(alpha_dim, 1, bias=False)

    def forward(self, M, x, mask=None):
        """
        M -> (seq_len, batch, mem_dim)
        x -> (batch, cand_dim) cand_dim == mem_dim?
        mask -> (batch, seq_len)
        """
        if type(mask)==type(None):
            mask = torch.ones(M.size(1), M.size(0)).type(M.type())

        if self.att_type=='dot':
            M_ = M.permute(1,2,0)
            x_ = x.unsqueeze(1)
            alpha = F.softmax(torch.bmm(x_, M_), dim=2)
        elif self.att_type=='general':
            M_ = M.permute(1,2,0)
            x_ = self.transform(x).unsqueeze(1)
            alpha = F.softmax(torch.bmm(x_, M_), dim=2)
        elif self.att_type=='general2':
            M_ = M.permute(1,2,0)
            x_ = self.transform(x).unsqueeze(1)
            mask_ = mask.unsqueeze(2).repeat(1, 1, self.mem_dim).transpose(1, 2)
            M_ = M_ * mask_
            alpha_ = torch.bmm(x_, M_)*mask.unsqueeze(1)
            alpha_ = torch.tanh(alpha_)
            alpha_ = F.softmax(alpha_, dim=2)
            alpha_masked = alpha_*mask.unsqueeze(1)
            alpha_sum = torch.sum(alpha_masked, dim=2, keepdim=True)
            alpha = alpha_masked/alpha_sum
        else:
            M_ = M.transpose(0,1)
            x_ = x.unsqueeze(1).expand(-1,M.size()[0],-1)
            M_x_ = torch.cat([M_,x_],2)
            mx_a = F.tanh(self.transform(M_x_))
            alpha = F.softmax(self.vector_prod(mx_a),1).transpose(1,2)

        attn_pool = torch.bmm(alpha, M.transpose(0,1))[:,0,:]
        return attn_pool, alpha


class Attention(nn.Module):
    def __init__(self, embed_dim, hidden_dim=None, out_dim=None, n_head=1, score_function='dot_product', dropout=0):
        ''' Attention Mechanism
        :param embed_dim:
        :param hidden_dim:
        :param out_dim:
        :param n_head: num of head (Multi-Head Attention)
        :param score_function: scaled_dot_product / mlp (concat) / bi_linear (general dot)
        :return (?, q_len, out_dim,)
        '''
        super(Attention, self).__init__()
        if hidden_dim is None:
            hidden_dim = embed_dim // n_head
        if out_dim is None:
            out_dim = embed_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_head = n_head
        self.score_function = score_function
        self.w_k = nn.Linear(embed_dim, n_head * hidden_dim)
        self.w_q = nn.Linear(embed_dim, n_head * hidden_dim)
        self.proj = nn.Linear(n_head * hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        if score_function == 'mlp':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim*2))
        elif self.score_function == 'bi_linear':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        else:
            self.register_parameter('weight', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.hidden_dim)
        if self.weight is not None:
            self.weight.data.uniform_(-stdv, stdv)

    def forward(self, k, q):
        if len(q.shape) == 2:
            q = torch.unsqueeze(q, dim=1)
        if len(k.shape) == 2:
            k = torch.unsqueeze(k, dim=1)
        mb_size = k.shape[0]
        k_len = k.shape[1]
        q_len = q.shape[1]
        # k: (?, k_len, embed_dim,)
        # q: (?, q_len, embed_dim,)
        # kx: (n_head*?, k_len, hidden_dim)
        # qx: (n_head*?, q_len, hidden_dim)
        # score: (n_head*?, q_len, k_len,)
        # output: (?, q_len, out_dim,)
        kx = self.w_k(k).view(mb_size, k_len, self.n_head, self.hidden_dim)
        kx = kx.permute(2, 0, 1, 3).contiguous().view(-1, k_len, self.hidden_dim)
        qx = self.w_q(q).view(mb_size, q_len, self.n_head, self.hidden_dim)
        qx = qx.permute(2, 0, 1, 3).contiguous().view(-1, q_len, self.hidden_dim)
        if self.score_function == 'dot_product':
            kt = kx.permute(0, 2, 1)
            score = torch.bmm(qx, kt)
        elif self.score_function == 'scaled_dot_product':
            kt = kx.permute(0, 2, 1)
            qkt = torch.bmm(qx, kt)
            score = torch.div(qkt, math.sqrt(self.hidden_dim))
        elif self.score_function == 'mlp':
            kxx = torch.unsqueeze(kx, dim=1).expand(-1, q_len, -1, -1)
            qxx = torch.unsqueeze(qx, dim=2).expand(-1, -1, k_len, -1)
            kq = torch.cat((kxx, qxx), dim=-1)
            score = torch.tanh(torch.matmul(kq, self.weight))
        elif self.score_function == 'bi_linear':
            qw = torch.matmul(qx, self.weight)
            kt = kx.permute(0, 2, 1)
            score = torch.bmm(qw, kt)
        else:
            raise RuntimeError('invalid score_function')
        score = F.softmax(score, dim=0)
        output = torch.bmm(score, kx)
        output = torch.cat(torch.split(output, mb_size, dim=0), dim=-1)
        output = self.proj(output)
        output = self.dropout(output)
        return output, score


class MaskedEdgeAttention(nn.Module):

    def __init__(self, input_dim, max_seq_len, no_cuda):
        """
        Method to compute the edge weights, as in Equation 1. in the paper.
        attn_type = 'attn1' refers to the equation in the paper.
        For slightly different attention mechanisms refer to attn_type = 'attn2' or attn_type = 'attn3'
        """

        super(MaskedEdgeAttention, self).__init__()

        self.input_dim = input_dim
        self.max_seq_len = max_seq_len
        self.scalar = nn.Linear(self.input_dim, self.max_seq_len, bias=False)
        self.matchatt = MatchingAttention(self.input_dim, self.input_dim, att_type='general2')
        self.simpleatt = SimpleAttention(self.input_dim)
        self.att = Attention(self.input_dim, score_function='mlp')
        self.no_cuda = no_cuda

    def forward(self, M, lengths, edge_ind):
        """
        M -> (seq_len, batch, vector)
        lengths -> length of the sequences in the batch
        edge_idn -> edge_idn是边的index的集合
        """
        attn_type = 'attn1'

        if attn_type == 'attn1':

            device = M.device
            scale = self.scalar(M)
            alpha = F.softmax(scale, dim=0).permute(1, 2, 0)

            mask = torch.zeros_like(alpha, device=device)
            mask_copy = torch.zeros_like(alpha, device=device)

            edge_ind_ = []
            for batch_index, batch_edges in enumerate(edge_ind):
                for src, dst in batch_edges:
                    edge_ind_.append([batch_index, src, dst])

            if len(edge_ind_) == 0:
                return torch.zeros_like(alpha, device=device)

            edge_ind_ = torch.tensor(edge_ind_, dtype=torch.long, device=device).t()
            idx_b, idx_i, idx_j = edge_ind_[0], edge_ind_[1], edge_ind_[2]
            valid = (idx_b >= 0) & (idx_b < alpha.size(0)) & \
                    (idx_i >= 0) & (idx_i < alpha.size(1)) & \
                    (idx_j >= 0) & (idx_j < alpha.size(2))
            if not valid.all():
                idx_b = idx_b[valid]
                idx_i = idx_i[valid]
                idx_j = idx_j[valid]

            mask[idx_b, idx_i, idx_j] = 1
            mask_copy[idx_b, idx_i, idx_j] = 1

            masked_alpha = alpha * mask
            _sums = masked_alpha.sum(-1, keepdim=True).clamp_min(1e-9)
            scores = masked_alpha.div(_sums) * mask_copy
            return scores

        elif attn_type == 'attn2':
            scores = torch.zeros(M.size(1), self.max_seq_len, self.max_seq_len, requires_grad=True)

            if not self.no_cuda:
                scores = scores.cuda()

            for j in range(M.size(1)):

                ei = np.array(edge_ind[j])

                for node in range(lengths[j]):
                    neighbour = ei[ei[:, 0] == node, 1]

                    M_ = M[neighbour, j, :].unsqueeze(1)
                    t = M[node, j, :].unsqueeze(0)
                    _, alpha_ = self.simpleatt(M_, t)
                    scores[j, node, neighbour] = alpha_

        elif attn_type == 'attn3':
            scores = torch.zeros(M.size(1), self.max_seq_len, self.max_seq_len, requires_grad=True)

            if not self.no_cuda:
                scores = scores.cuda()

            for j in range(M.size(1)):

                ei = np.array(edge_ind[j])

                for node in range(lengths[j]):
                    neighbour = ei[ei[:, 0] == node, 1]

                    M_ = M[neighbour, j, :].unsqueeze(1).transpose(0, 1)
                    t = M[node, j, :].unsqueeze(0).unsqueeze(0).repeat(len(neighbour), 1, 1).transpose(0, 1)
                    _, alpha_ = self.att(M_, t)
                    scores[j, node, neighbour] = alpha_[0, :, 0]

        return scores


def pad(tensor, length, no_cuda):
    if isinstance(tensor, Variable):
        var = tensor
        if length > var.size(0):
            if not no_cuda:
                return torch.cat([var, torch.zeros(length - var.size(0), *var.size()[1:]).cuda()])
            else:
                return torch.cat([var, torch.zeros(length - var.size(0), *var.size()[1:])])
        else:
            return var
    else:
        if length > tensor.size(0):
            if not no_cuda:
                return torch.cat([tensor, torch.zeros(length - tensor.size(0), *tensor.size()[1:]).cuda()])
            else:
                return torch.cat([tensor, torch.zeros(length - tensor.size(0), *tensor.size()[1:])])
        else:
            return tensor


def edge_perms(l, window_past, window_future):
    """快速生成一个会话中所有允许的边对。"""
    if l == 0:
        return []

    if window_past == -1 and window_future == -1:
        return [(i, j) for i in range(l) for j in range(l)]

    edge_pairs = []
    for i in range(l):
        start = 0 if window_past == -1 else max(0, i - window_past)
        end = l if window_future == -1 else min(l, i + window_future + 1)
        for j in range(start, end):
            edge_pairs.append((i, j))
    return edge_pairs


def simple_batch_graphify(features, lengths, no_cuda):
    edge_index, edge_norm, edge_type, node_features = [], [], [], []
    batch_size = features.size(1)

    for j in range(batch_size):
        node_features.append(features[:lengths[j], j, :])

    node_features = torch.cat(node_features, dim=0)

    if not no_cuda:
        node_features = node_features.cuda()

    return node_features, None, None, None, None


def batch_graphify(features, qmask, lengths, window_past, window_future, edge_type_mapping, att_model, no_cuda):
    edge_index = []
    edge_norm = []
    edge_type = []
    node_features = []
    batch_size = features.size(0)
    assert len(lengths) == batch_size, \
        f"batch_graphify lengths mismatch: expected {batch_size}, got {len(lengths)}"
    length_sum = 0
    edge_index_lengths = []

    # 预先计算所有边对
    edge_ind = [edge_perms(lengths[j], window_past, window_future) for j in range(batch_size)]
    features = features.permute(1, 0, 2)
    device = features.device
    scores = att_model(features, lengths, edge_ind)

    for j in range(batch_size):
        seq_len = lengths[j]
        node_features.append(features[:seq_len, j, :])

        perms1 = edge_ind[j]
        perms2 = [(src + length_sum, dst + length_sum) for src, dst in perms1]
        length_sum += seq_len

        edge_index_lengths.append(len(perms1))

        speaker_ids = torch.argmax(qmask[:seq_len, j, :], dim=-1)

        for (src, dst), (src_idx, dst_idx) in zip(perms1, perms2):
            edge_index.append([src_idx, dst_idx])
            edge_norm.append(scores[j, src, dst])

            speaker0 = speaker_ids[src].item()
            speaker1 = speaker_ids[dst].item()
            edge_key = f"{speaker0}{speaker1}{'0' if src < dst else '1'}"
            edge_type.append(edge_type_mapping[edge_key])

    if len(node_features) > 0:
        node_features = torch.cat(node_features, dim=0)
    else:
        node_features = torch.empty(0, features.size(-1), device=device)

    if len(edge_index) > 0:
        edge_index = torch.tensor(edge_index, dtype=torch.long, device=device).t()
        edge_norm = torch.stack(edge_norm)
        edge_type = torch.tensor(edge_type, dtype=torch.long, device=device)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        edge_norm = torch.empty((0,), dtype=torch.float, device=device)
        edge_type = torch.empty((0,), dtype=torch.long, device=device)

    return node_features, edge_index, edge_norm, edge_type, edge_index_lengths


def attentive_node_features(emotions, seq_lengths, umask, matchatt_layer, no_cuda):
    """
    Method to obtain attentive node features over the graph convoluted features, as in Equation 4, 5, 6. in the paper.
    """

    input_conversation_length = torch.tensor(seq_lengths)
    start_zero = input_conversation_length.data.new(1).zero_()

    if not no_cuda:
        input_conversation_length = input_conversation_length.cuda()
        start_zero = start_zero.cuda()

    max_len = max(seq_lengths)

    start = torch.cumsum(torch.cat((start_zero, input_conversation_length[:-1])), 0)

    emotions = torch.stack([pad(emotions.narrow(0, s, l), max_len, no_cuda)
                            for s, l in zip(start.data.tolist(),
                                            input_conversation_length.data.tolist())], 0).transpose(0, 1)

    alpha, alpha_f, alpha_b = [], [], []
    att_emotions = []

    for t in emotions:
        att_em, alpha_ = matchatt_layer(emotions, t, mask=umask)
        att_emotions.append(att_em.unsqueeze(0))
        alpha.append(alpha_[:, 0, :])

    att_emotions = torch.cat(att_emotions, dim=0)

    return att_emotions


def classify_node_features(emotions, seq_lengths, umask, matchatt_layer, linear_layer, dropout_layer, smax_fc_layer,
                           nodal_attn, avec, no_cuda):
    if nodal_attn:

        emotions = attentive_node_features(emotions, seq_lengths, umask, matchatt_layer, no_cuda)
        hidden = F.relu(linear_layer(emotions))
        hidden = dropout_layer(hidden)
        hidden = smax_fc_layer(hidden)

        if avec:
            return torch.cat([hidden[:, j, :][:seq_lengths[j]] for j in range(len(seq_lengths))])

        log_prob = F.log_softmax(hidden, 2)
        log_prob = torch.cat([log_prob[:, j, :][:seq_lengths[j]] for j in range(len(seq_lengths))])
        return log_prob

    else:

        hidden = F.relu(linear_layer(emotions))
        hidden = dropout_layer(hidden)
        hidden = smax_fc_layer(hidden)

        if avec:
            return hidden

        log_prob = F.log_softmax(hidden, 1)
        return log_prob


class ComplexLinear(nn.Module):
    """Complex-valued linear layer for spectral feature transforms."""

    def __init__(self, in_features, out_features, bias=True):
        super(ComplexLinear, self).__init__()
        self.real = nn.Parameter(torch.Tensor(out_features, in_features))
        self.imag = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias_real = nn.Parameter(torch.Tensor(out_features))
            self.bias_imag = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias_real', None)
            self.register_parameter('bias_imag', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.real.size(1))
        self.real.data.uniform_(-stdv, stdv)
        self.imag.data.uniform_(-stdv, stdv)
        if self.bias_real is not None:
            self.bias_real.data.zero_()
            self.bias_imag.data.zero_()

    def forward(self, x):
        x_real = x.real
        x_imag = x.imag
        real = F.linear(x_real, self.real, self.bias_real) - F.linear(x_imag, self.imag, self.bias_imag)
        imag = F.linear(x_imag, self.real, self.bias_imag) + F.linear(x_real, self.imag, self.bias_real)
        return torch.complex(real, imag)


class SpectralBlock(nn.Module):
    """A residual spectral transform block for Fourier-domain features."""

    def __init__(self, dim, dropout=0.0, sparsity_threshold=0.01):
        super(SpectralBlock, self).__init__()
        self.linear = ComplexLinear(dim, dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.sparsity_threshold = sparsity_threshold

    def forward(self, x):
        z = self.linear(x)
        z = torch.complex(F.relu(z.real), F.relu(z.imag))
        if self.dropout is not None:
            z = torch.complex(self.dropout(z.real), self.dropout(z.imag))
        z = torch.complex(F.softshrink(z.real, lambd=self.sparsity_threshold),
                          F.softshrink(z.imag, lambd=self.sparsity_threshold))
        return x + z


class FGN(nn.Module):
    def __init__(self, pre_length, embed_size,
                 feature_size, seq_length, hidden_size, hard_thresholding_fraction=1, hidden_size_factor=1, sparsity_threshold=0.01,
                 dropout=0.1, num_layers=3):
        super().__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.number_frequency = 1
        self.pre_length = pre_length
        self.feature_size = feature_size
        self.seq_length = seq_length
        self.frequency_size = self.embed_size // self.number_frequency
        self.hidden_size_factor = hidden_size_factor
        self.sparsity_threshold = sparsity_threshold
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.scale = 0.02
        self.embeddings = nn.Parameter(torch.randn(1, self.embed_size))

        self.spectral_blocks = nn.ModuleList([
            SpectralBlock(self.frequency_size, dropout=dropout, sparsity_threshold=sparsity_threshold)
            for _ in range(num_layers)
        ])

        self.embeddings_10 = nn.Parameter(torch.randn(self.seq_length, 8))
        self.fc = nn.Sequential(
            nn.Linear(self.embed_size * 8, 64),
            nn.LeakyReLU(),
            nn.Linear(64, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.pre_length)
        )

    def tokenEmb(self, x):
        x = x.unsqueeze(2)
        return x * self.embeddings

    def fourierGC(self, x):
        for block in self.spectral_blocks:
            x = block(x)
        return x

    def forward(self, x):
        B, N = x.shape
        x = self.tokenEmb(x)
        x = torch.fft.rfft(x, dim=1, norm='ortho')
        x = x.reshape(B, (N) // 2 + 1, self.frequency_size)

        bias = x
        x = self.fourierGC(x)
        x = x + bias

        x = x.reshape(B, (N) // 2 + 1, self.embed_size)
        x = torch.fft.irfft(x, n=N, dim=1, norm='ortho')
        x = x.reshape(B, N, self.embed_size)

        x = torch.matmul(x, self.embeddings_10)
        x = x.reshape(B, -1)
        x = self.fc(x)
        return x


class GraphNetwork(torch.nn.Module):
    def __init__(self, num_features, num_classes, num_relations, hidden_size=64, dropout=0.5,
                 no_cuda=False, return_feature=False):
        super(GraphNetwork, self).__init__()

        self.return_feature = return_feature
        self.no_cuda = no_cuda
        self.conv1 = RGCNConv(num_features, hidden_size, num_relations, num_bases=30)
        self.conv2 = FGN(hidden_size, hidden_size, hidden_size, hidden_size, hidden_size)
        if not self.return_feature:
            self.matchatt = MatchingAttention(num_features + hidden_size, num_features + hidden_size,
                                              att_type='general2')
            self.linear = nn.Linear(num_features + hidden_size, hidden_size)
            self.dropout = nn.Dropout(dropout)
            self.smax_fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, edge_index, edge_norm, edge_type, seq_lengths, umask, nodal_attn, avec):
        out = self.conv1(x, edge_index, edge_type)

        if edge_norm is not None and edge_norm.numel() > 0:
            dst = edge_index[1]
            node_edge_score = torch.zeros(x.size(0), device=x.device).scatter_add_(0, dst, edge_norm)
            node_degree = torch.bincount(dst, minlength=x.size(0)).unsqueeze(-1).clamp_min(1.0)
            node_scaling = node_edge_score.unsqueeze(-1) / node_degree
            out = out * (1.0 + node_scaling)

        out = self.conv2(out)
        emotions = torch.cat([x, out], dim=-1)
        if self.return_feature:
            return emotions
        log_prob = classify_node_features(emotions, seq_lengths, umask, self.matchatt, self.linear, self.dropout,
                                          self.smax_fc, nodal_attn, avec, self.no_cuda)
        return log_prob

class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=512):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                              -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x, speaker_emb):
        L = x.size(1)
        pos_emb = self.pe[:, :L]
        x = x + pos_emb + speaker_emb
        return x


class CrossModalAttention(nn.Module):
    """
    跨模态注意力机制：允许不同模态之间进行特征交互和增强
    """
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super(CrossModalAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: (batch_size, seq_len, embed_dim)
            key: (batch_size, seq_len, embed_dim)
            value: (batch_size, seq_len, embed_dim)
            mask: (batch_size, seq_len) - optional mask for padding
        Returns:
            output: (batch_size, seq_len, embed_dim)
            attention_weights: (batch_size, num_heads, seq_len, seq_len)
        """
        batch_size, seq_len, _ = query.size()

        # Linear transformations and reshape
        Q = self.q_linear(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(key).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(value).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        context = torch.matmul(attention_weights, V)

        # Concatenate heads and put through final linear layer
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        output = self.out_linear(context)

        # Residual connection and layer norm
        output = self.layer_norm(output + query)

        return output, attention_weights


class ModalAdaptiveWeight(nn.Module):
    """
    模态自适应权重学习：学习每个模态的重要性权重
    """
    def __init__(self, num_modals, embed_dim):
        super(ModalAdaptiveWeight, self).__init__()
        self.num_modals = num_modals
        self.embed_dim = embed_dim

        # 权重预测网络
        self.weight_net = nn.Sequential(
            nn.Linear(embed_dim * num_modals, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, num_modals),
            nn.Softmax(dim=-1)
        )

        # 模态特定变换
        self.modal_transforms = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim) for _ in range(num_modals)
        ])

    def forward(self, modal_features):
        """
        Args:
            modal_features: list of tensors, each (batch_size, seq_len, embed_dim)
        Returns:
            weighted_features: (batch_size, seq_len, embed_dim)
            weights: (batch_size, num_modals)
        """
        # 拼接所有模态特征用于权重预测
        concat_features = torch.cat(modal_features, dim=-1)  # (batch_size, seq_len, embed_dim * num_modals)

        # 预测权重 (在序列维度上平均)
        avg_features = torch.mean(concat_features, dim=1)  # (batch_size, embed_dim * num_modals)
        weights = self.weight_net(avg_features)  # (batch_size, num_modals)

        # 应用模态特定变换
        transformed_features = []
        for i, features in enumerate(modal_features):
            transformed = self.modal_transforms[i](features)
            transformed_features.append(transformed)

        # 加权融合
        weighted_features = torch.zeros_like(transformed_features[0])
        for i, features in enumerate(transformed_features):
            weight = weights[:, i].unsqueeze(-1).unsqueeze(-1)  # (batch_size, 1, 1)
            weighted_features += weight * features

        return weighted_features, weights


class EnhancedModalFusion(nn.Module):
    """
    增强的模态融合模块：结合跨模态注意力和自适应权重
    """
    def __init__(self, embed_dim, num_modals, num_heads=8, dropout=0.1):
        super(EnhancedModalFusion, self).__init__()
        self.embed_dim = embed_dim
        self.num_modals = num_modals

        # 跨模态注意力层
        self.cross_attention_layers = nn.ModuleList([
            CrossModalAttention(embed_dim, num_heads, dropout)
            for _ in range(num_modals)
        ])

        # 模态自适应权重
        self.adaptive_weight = ModalAdaptiveWeight(num_modals, embed_dim)

        # 最终融合层
        self.fusion_net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, modal_features, mask=None):
        """
        Args:
            modal_features: list of tensors, each (batch_size, seq_len, embed_dim)
            mask: (batch_size, seq_len) - optional mask
        Returns:
            fused_features: (batch_size, seq_len, embed_dim)
            attention_weights: list of attention weight tensors
            modal_weights: (batch_size, num_modals)
        """
        enhanced_features = []
        attention_weights = []

        # 对每个模态，使用其他模态进行增强
        for i in range(self.num_modals):
            target_modal = modal_features[i]

            # 收集其他模态作为上下文
            context_modals = [modal_features[j] for j in range(self.num_modals) if j != i]
            if context_modals:
                # 使用其他模态的平均作为上下文
                context = torch.stack(context_modals, dim=0).mean(dim=0)

                # 跨模态注意力
                enhanced, attn_weights = self.cross_attention_layers[i](target_modal, context, context, mask)
                attention_weights.append(attn_weights)
            else:
                enhanced = target_modal
                attention_weights.append(None)

            enhanced_features.append(enhanced)

        # 自适应权重融合
        fused_features, modal_weights = self.adaptive_weight(enhanced_features)

        # 最终融合
        fused_features = self.fusion_net(fused_features)

        return fused_features, attention_weights, modal_weights


class EnhancedEmotionClassifier(nn.Module):
    """
    增强的情感分类器：支持多层架构、残差连接和多种正则化技术
    """
    def __init__(self, input_dim, n_classes, hidden_dims=[512, 256], dropout=0.5,
                 use_residual=True, use_batch_norm=True, activation='relu'):
        super(EnhancedEmotionClassifier, self).__init__()

        self.input_dim = input_dim
        self.n_classes = n_classes
        self.hidden_dims = hidden_dims
        self.use_residual = use_residual
        self.use_batch_norm = use_batch_norm

        # 激活函数选择
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.1)
        else:
            self.activation = nn.ReLU()

        # 构建多层分类器
        layers = []
        current_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            # 线性层
            layers.append(nn.Linear(current_dim, hidden_dim))

            # 批归一化
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            # 激活函数
            layers.append(self.activation)

            # Dropout
            layers.append(nn.Dropout(dropout))

            # 残差连接准备
            if use_residual and current_dim == hidden_dim:
                self.residual_layer = nn.Identity()
            elif use_residual and i == 0:
                self.residual_proj = nn.Linear(input_dim, hidden_dims[-1])

            current_dim = hidden_dim

        # 输出层
        layers.append(nn.Linear(current_dim, n_classes))

        self.classifier = nn.Sequential(*layers)

        # 温度缩放参数（用于校准）
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, x, return_logits=False):
        """
        Args:
            x: 输入特征 (batch_size, seq_len, input_dim) 或 (batch_size, input_dim)
            return_logits: 是否返回logits而不是概率
        Returns:
            分类概率或logits
        """
        # 处理序列输入
        if x.dim() == 3:
            batch_size, seq_len, feature_dim = x.size()
            x = x.view(-1, feature_dim)  # (batch_size * seq_len, feature_dim)

        # 残差连接
        if self.use_residual and hasattr(self, 'residual_proj'):
            residual = self.residual_proj(x)

        # 分类器前向传播
        logits = self.classifier(x)

        # 温度缩放
        scaled_logits = logits / self.temperature

        if return_logits:
            return scaled_logits

        # 返回概率分布
        return F.softmax(scaled_logits, dim=-1)


class FocalLoss(nn.Module):
    """
    焦点损失：用于处理类别不平衡问题
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

        if alpha is not None:
            if isinstance(alpha, (list, tuple)):
                self.alpha = torch.tensor(alpha)
            elif isinstance(alpha, torch.Tensor):
                self.alpha = alpha
            else:
                self.alpha = torch.ones(alpha)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: 预测logits (N, C)
            targets: 目标标签 (N,)
        """
        # 计算交叉熵损失
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # 计算预测概率
        pt = torch.exp(-ce_loss)

        # 计算焦点权重
        focal_weight = (1 - pt) ** self.gamma

        # 应用类别权重
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_weight = self.alpha[targets]
            focal_weight = focal_weight * alpha_weight

        # 应用权重
        loss = focal_weight * ce_loss

        # 损失聚合
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class LabelSmoothingCrossEntropy(nn.Module):
    """
    标签平滑交叉熵损失
    """
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        confidence = 1. - self.smoothing
        log_probs = F.log_softmax(pred, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class MultiTaskEmotionClassifier(nn.Module):
    """
    多任务情感分类器：同时预测多个相关任务
    """
    def __init__(self, input_dim, n_classes, hidden_dims=[512, 256], dropout=0.5,
                 use_auxiliary=True, auxiliary_weight=0.3):
        super(MultiTaskEmotionClassifier, self).__init__()

        self.input_dim = input_dim
        self.n_classes = n_classes
        self.use_auxiliary = use_auxiliary
        self.auxiliary_weight = auxiliary_weight

        # 主分类器
        self.main_classifier = EnhancedEmotionClassifier(
            input_dim, n_classes, hidden_dims, dropout
        )

        # 辅助任务：情感强度预测（回归任务）
        if use_auxiliary:
            self.intensity_predictor = nn.Sequential(
                nn.Linear(input_dim, hidden_dims[0] // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dims[0] // 2, 1),
                nn.Sigmoid()  # 输出0-1之间的强度值
            )

        # 置信度估计器
        self.confidence_estimator = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0] // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0] // 2, 1),
            nn.Sigmoid()  # 输出0-1之间的置信度
        )

    def forward(self, x, return_logits=False, return_auxiliary=False):
        """
        Args:
            x: 输入特征
            return_logits: 是否返回logits而不是概率
            return_auxiliary: 是否返回辅助任务输出
        Returns:
            主分类概率或logits，(可选)辅助输出
        """
        # 主分类
        main_output = self.main_classifier(x, return_logits=return_logits)

        if not return_auxiliary:
            return main_output

        # 辅助任务
        auxiliary_outputs = {}

        if self.use_auxiliary:
            intensity = self.intensity_predictor(x)
            auxiliary_outputs['intensity'] = intensity

        confidence = self.confidence_estimator(x)
        auxiliary_outputs['confidence'] = confidence

        return main_output, auxiliary_outputs

    def compute_loss(self, main_logits, targets, auxiliary_outputs=None, class_weights=None):
        """
        计算多任务损失
        """
        # 主分类损失
        if class_weights is not None:
            main_loss = F.cross_entropy(main_logits, targets, weight=class_weights)
        else:
            main_loss = F.cross_entropy(main_logits, targets)

        total_loss = main_loss

        # 辅助任务损失
        if auxiliary_outputs is not None:
            # 这里可以添加辅助任务的监督信号
            # 例如，如果有强度标签或置信度标签
            pass

        return total_loss


class DialogueGCNModel(nn.Module):

    def __init__(self, base_model, D_m, D_e, D_m_v, D_m_a, graph_hidden_size, n_speakers, max_seq_len,
                 window_past, window_future,
                 n_classes=7, dropout=0.5,
                 nodal_attention=True, avec=False,
                 no_cuda=False, graph_type='relation', use_topic=False, alpha=0.2, multiheads=6,
                 graph_construct='direct', use_residue=True,
                 dynamic_edge_w=False, hidden_dim=200, modals='avl', att_type='gated', av_using_lstm=False,
                dataset='IEMOCAP',
                 use_speaker=True, use_modal=False):

        super(DialogueGCNModel, self).__init__()

        self.base_model = base_model
        self.avec = avec
        self.no_cuda = no_cuda
        self.graph_type = graph_type
        self.alpha = alpha
        self.multiheads = multiheads
        self.graph_construct = graph_construct
        self.use_topic = use_topic
        self.dropout = dropout
        self.use_residue = use_residue
        self.dynamic_edge_w = dynamic_edge_w
        self.return_feature = True
        self.modals = [x for x in modals]  # a, v, l
        self.use_speaker = use_speaker
        self.use_modal = use_modal
        self.n_speakers = n_speakers
        if self.n_speakers == 2:
            padding_idx = 2
        if self.n_speakers == 9:
            padding_idx = 9
        self.speaker_embeddings = nn.Embedding(n_speakers + 1, hidden_dim, padding_idx)
        self.att_type = att_type
        if self.att_type == 'gated' or self.att_type == 'concat_subsequently':
            self.multi_modal = True
            self.av_using_lstm = av_using_lstm
        else:
            self.multi_modal = False
        self.use_bert_seq = False
        self.dataset = dataset


        self.linear_a = nn.Linear(D_m_a, hidden_dim)
        self.linear_v = nn.Linear(D_m_v, hidden_dim)
        self.linear_t = nn.Linear(D_m, hidden_dim)

        self.lstm_t = nn.LSTM(input_size=hidden_dim, hidden_size=D_e, num_layers=2, bidirectional=True,
                            dropout=dropout, batch_first=True)
        self.lstm_a = nn.LSTM(input_size=hidden_dim, hidden_size=D_e, num_layers=2, bidirectional=True,
                            dropout=dropout, batch_first=True)
        self.lstm_v = nn.LSTM(input_size=hidden_dim, hidden_size=D_e, num_layers=2, bidirectional=True,
                            dropout=dropout, batch_first=True)

        self.pos_emb_a = PositionalEncoding(hidden_dim)
        self.pos_emb_v = PositionalEncoding(hidden_dim)
        self.pos_emb_t = PositionalEncoding(hidden_dim)

        n_relations = 2 * n_speakers ** 2
        self.window_past = window_past
        self.window_future = window_future

        self.att_model = MaskedEdgeAttention(2 * D_e, max_seq_len, self.no_cuda)
        self.nodal_attention = nodal_attention

        self.graph_net_a = GraphNetwork(2 * D_e, n_classes, n_relations, graph_hidden_size,
                                            dropout, self.no_cuda, self.return_feature)
        self.graph_net_v = GraphNetwork(2 * D_e, n_classes, n_relations, graph_hidden_size,
                                            dropout, self.no_cuda, self.return_feature)
        self.graph_net_l = GraphNetwork(2 * D_e, n_classes, n_relations, graph_hidden_size,
                                            dropout, self.no_cuda, self.return_feature)
        print("construct fourier graph")

        edge_type_mapping = {}
        for j in range(n_speakers):
            for k in range(n_speakers):
                edge_type_mapping[str(j) + str(k) + '0'] = len(edge_type_mapping)
                edge_type_mapping[str(j) + str(k) + '1'] = len(edge_type_mapping)

        self.edge_type_mapping = edge_type_mapping

        # 使用增强的模态融合模块替代原来的门控注意力
        if self.multi_modal and len(self.modals) > 1:
            self.enhanced_fusion = EnhancedModalFusion(2 * D_e + graph_hidden_size, len(self.modals),
                                                     num_heads=8, dropout=dropout)
        else:
            self.gatedatt = MMGatedAttention(2 * D_e + graph_hidden_size, graph_hidden_size, att_type='general')

        self.dropout_ = nn.Dropout(self.dropout)
        if self.att_type == 'concat_subsequently':
            self.smax_fc = nn.Linear(300 * len(self.modals), n_classes)
        elif self.att_type == 'gated':
            if len(self.modals) == 3:
                self.smax_fc = nn.Linear(100 * len(self.modals), n_classes)
            else:
                self.smax_fc = nn.Linear(100, n_classes)
        elif self.att_type == 'enhanced_fusion':
            # 增强融合的输出维度
            self.smax_fc = nn.Linear(2 * D_e + graph_hidden_size, n_classes)
        else:
            self.smax_fc = nn.Linear(2 * D_e + graph_hidden_size, n_classes)

        # 使用增强的情感分类器
        classifier_input_dim = 2 * D_e + graph_hidden_size
        self.enhanced_classifier = MultiTaskEmotionClassifier(
            input_dim=classifier_input_dim,
            n_classes=n_classes,
            hidden_dims=[512, 256],
            dropout=dropout,
            use_auxiliary=True
        )

        # 保留原有的简单输出层作为备选
        self.t_output_layer = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * D_e + graph_hidden_size, n_classes)
        )
        self.a_output_layer = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * D_e + graph_hidden_size, n_classes)
        )
        self.v_output_layer = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * D_e + graph_hidden_size, n_classes)
        )

    def forward(self, textf, acouf, visuf, qmask, umask, seq_lengths):
        spk_idx = torch.argmax(qmask.permute(1, 0, 2), -1)
        origin_spk_idx = spk_idx
        if self.n_speakers == 2:
            for i, x in enumerate(seq_lengths):
                spk_idx[i, x:] = (2 * torch.ones(origin_spk_idx[i].size(0) - x)).int().cuda()
        if self.n_speakers == 9:
            for i, x in enumerate(seq_lengths):
                spk_idx[i, x:] = (9 * torch.ones(origin_spk_idx[i].size(0) - x)).int().cuda()
        spk_embeddings = self.speaker_embeddings(spk_idx)

        textf = textf.permute(1, 2, 0).transpose(1, 2)
        acouf = acouf.permute(1, 2, 0).transpose(1, 2)
        visuf = visuf.permute(1, 2, 0).transpose(1, 2)

        textf = self.pos_emb_t(self.linear_t(textf), spk_embeddings)
        acouf = self.pos_emb_t(self.linear_a(acouf), spk_embeddings)
        visuf = self.pos_emb_t(self.linear_v(visuf), spk_embeddings)

        emotions_a, _ = self.lstm_t(textf)
        emotions_v, _ = self.lstm_a(acouf)
        emotions_t, _ = self.lstm_v(visuf)


        features_a, edge_index, edge_norm, edge_type, edge_index_lengths = batch_graphify(emotions_a,
                                                                                              qmask,
                                                                                              seq_lengths,
                                                                                              self.window_past,
                                                                                              self.window_future,
                                                                                              self.edge_type_mapping,
                                                                                              self.att_model,
                                                                                              self.no_cuda)

        features_v, edge_index, edge_norm, edge_type, edge_index_lengths = batch_graphify(emotions_v,
                                                                                          qmask,
                                                                                          seq_lengths,
                                                                                          self.window_past,
                                                                                          self.window_future,
                                                                                          self.edge_type_mapping,
                                                                                          self.att_model,
                                                                                          self.no_cuda)

        features_l, edge_index, edge_norm, edge_type, edge_index_lengths = batch_graphify(emotions_t,
                                                                                          qmask,
                                                                                          seq_lengths,
                                                                                          self.window_past,
                                                                                          self.window_future,
                                                                                          self.edge_type_mapping,
                                                                                          self.att_model,
                                                                                          self.no_cuda)


        emotions_a = self.graph_net_a(features_a, edge_index, edge_norm, edge_type, seq_lengths, umask,
                                      self.nodal_attention, self.avec)


        emotions_v = self.graph_net_v(features_v, edge_index, edge_norm, edge_type, seq_lengths, umask,
                                      self.nodal_attention, self.avec)


        emotions_t = self.graph_net_l(features_l, edge_index, edge_norm, edge_type, seq_lengths, umask,
                                      self.nodal_attention, self.avec)

        # 使用增强分类器进行预测
        t_final_out = self.enhanced_classifier(emotions_t, return_logits=True)
        a_final_out = self.enhanced_classifier(emotions_a, return_logits=True)
        v_final_out = self.enhanced_classifier(emotions_v, return_logits=True)

        # 获取概率分布
        t_log_prob = F.log_softmax(t_final_out, 1)
        a_log_prob = F.log_softmax(a_final_out, 1)
        v_log_prob = F.log_softmax(v_final_out, 1)

        # 获取多模态融合的概率
        _, auxiliary_t = self.enhanced_classifier(emotions_t, return_auxiliary=True)
        _, auxiliary_a = self.enhanced_classifier(emotions_a, return_auxiliary=True)
        _, auxiliary_v = self.enhanced_classifier(emotions_v, return_auxiliary=True)

        kl_t_log_prob = F.log_softmax(t_final_out, 1)
        kl_a_log_prob = F.log_softmax(a_final_out, 1)
        kl_v_log_prob = F.log_softmax(v_final_out, 1)

        # 使用增强的模态融合或传统的门控注意力
        if self.multi_modal and len(self.modals) > 1 and hasattr(self, 'enhanced_fusion'):
            # 准备模态特征列表
            modal_features = []
            if 'a' in self.modals:
                modal_features.append(emotions_a)
            if 'v' in self.modals:
                modal_features.append(emotions_v)
            if 'l' in self.modals:
                modal_features.append(emotions_t)

            # 创建mask用于跨模态注意力
            batch_size = emotions_a.size(0) if len(emotions_a.size()) > 0 else emotions_t.size(0)
            seq_len = max(seq_lengths)
            mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=emotions_t.device)
            for i, length in enumerate(seq_lengths):
                mask[i, :length] = 1

            # 增强模态融合
            emotions_feat, attention_weights, modal_weights = self.enhanced_fusion(modal_features, mask)
        else:
            # 传统门控注意力融合
            emotions_feat = self.gatedatt(emotions_a, emotions_v, emotions_t, self.modals)

        emotions_feat = self.dropout_(emotions_feat)

        # 使用增强分类器进行最终多模态分类
        final_out = self.enhanced_classifier(emotions_feat, return_logits=True)
        all_log_prob = F.log_softmax(final_out, 1)
        all_prob = F.softmax(final_out, 1)
        kl_all_prob = F.softmax(final_out, 1)

        # 获取辅助输出用于分析
        _, auxiliary_all = self.enhanced_classifier(emotions_feat, return_auxiliary=True)

        return t_log_prob, a_log_prob, v_log_prob, all_log_prob, all_prob, \
               kl_t_log_prob, kl_a_log_prob, kl_v_log_prob, kl_all_prob, edge_index, edge_norm, edge_type, edge_index_lengths
