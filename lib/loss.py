import torch
import torch.nn.functional as F


def kl_loss(code):
    return torch.mean(torch.pow(code, 2))


def pairwise_cosine_similarity(seqs_i, seqs_j):
    # seqs_i, seqs_j: [batch, statics, channel]
    n_statics = seqs_i.size(1)
    seqs_i_exp = seqs_i.unsqueeze(2).repeat(1, 1, n_statics, 1)
    seqs_j_exp = seqs_j.unsqueeze(1).repeat(1, n_statics, 1, 1)
    return F.cosine_similarity(seqs_i_exp, seqs_j_exp, dim=3)


def temporal_pairwise_cosine_similarity(seqs_i, seqs_j):
    # seqs_i, seqs_j: [batch, channel, time]
    seq_len = seqs_i.size(2)
    seqs_i_exp = seqs_i.unsqueeze(3).repeat(1, 1, 1, seq_len)
    seqs_j_exp = seqs_j.unsqueeze(2).repeat(1, 1, seq_len, 1)
    return F.cosine_similarity(seqs_i_exp, seqs_j_exp, dim=1)


def consecutive_cosine_similarity(seqs):
    # seqs: [batch, channel, time]
    seqs_roll = seqs.roll(shifts=1, dim=2)[1:]
    seqs = seqs[:-1]
    return F.cosine_similarity(seqs, seqs_roll)


def triplet_margin_loss(seqs_a, seqs_b, neg_range=(0.0, 0.5), margin=0.2):
    # seqs_a, seqs_b: [batch, channel, time]

    neg_start, neg_end = neg_range
    batch_size, _, seq_len = seqs_a.size()
    n_neg_all = seq_len ** 2
    n_neg = int(round(neg_end * n_neg_all))
    n_neg_discard = int(round(neg_start * n_neg_all))

    batch_size, _, seq_len = seqs_a.size()
    sim_aa = temporal_pairwise_cosine_similarity(seqs_a, seqs_a)
    sim_bb = temporal_pairwise_cosine_similarity(seqs_b, seqs_b)
    sim_ab = temporal_pairwise_cosine_similarity(seqs_a, seqs_b)
    sim_ba = sim_ab.transpose(1, 2)

    diff_ab = (sim_ab - sim_aa).reshape(batch_size, -1)
    diff_ba = (sim_ba - sim_bb).reshape(batch_size, -1)
    diff = torch.cat([diff_ab, diff_ba], dim=0)
    diff, _ = diff.topk(n_neg, dim=-1, sorted=True)
    diff = diff[:, n_neg_discard:]

    loss = diff + margin
    loss = loss.clamp(min=0.)
    loss = loss.mean()

    return loss