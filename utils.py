import torch
import torch.nn.functional as F
from torch import nn


def convert_pad_shape(pad_shape):
    l = pad_shape[::-1]
    pad_shape = [item for sublist in l for item in sublist]
    return pad_shape


def sequence_mask(x_length, max_len=None):
    if max_len is None:
        max_len = x_length.max()
    x = torch.arange(max_len, dtype=x_length.dtype, device=x_length.device)
    print(x.shape)
    return x.unsqueeze(0) < x_length.unsqueeze(1)


def pad(input_ele, mel_max_length=None):
    if mel_max_length:
        max_len = mel_max_length
    else:
        max_len = max([input_ele[i].size(0) for i in range(len(input_ele))])

    out_list = list()
    for i, batch in enumerate(input_ele):
        if len(batch.shape) == 1:
            one_batch_padded = F.pad(
                batch, (0, max_len - batch.size(0)), "constant", 0.0
            )
        elif len(batch.shape) == 2:
            one_batch_padded = F.pad(
                batch, (0, 0, 0, max_len - batch.size(0)), "constant", 0.0
            )
        out_list.append(one_batch_padded)
    out_padded = torch.stack(out_list)
    return out_padded


def word_level_pooling(src_seq, src_len, wb, src_w_len, reduce="mean"):
    print(src_seq.shape, src_len.shape, wb.shape, src_w_len.shape)
    """
    :param src_seq -- [batch_size, max_time, dim]
    :param src_len -- [batch_size,]
    :param wb -- [batch_size, max_time]
    :param src_w_len -- [batch_size,]
    """
    batch, device = [], src_seq.device
    for s, sl, w, wl in zip(src_seq, src_len, wb, src_w_len):
        print(s.shape, sl.shape, w.shape, wl.shape)
        m, split_size = s[:sl, :], list(w[:wl].int())
        print(m.shape, split_size)
        m = nn.utils.rnn.pad_sequence(torch.split(m, split_size, dim=0))
        if reduce == "sum":
            m = torch.sum(m, dim=0)  # [src_w_len, hidden]
        elif reduce == "mean":
            m = torch.div(torch.sum(m, dim=0), torch.tensor(
                split_size, device=device).unsqueeze(-1))  # [src_w_len, hidden]
        else:
            raise ValueError()
        batch.append(m)
    return pad(batch).to(device)


def word_to_phones(w_seq, w_l, p_w, p_l):
    """
    :param w_seq: [batch_size, max_words, hidden]
    :param w_l: [batch_size,]
    :param s_n: [batch_size,]
    :return:
    """
    batch, device = [], w_seq.device
    for w, wl, p_w in zip(w_seq, w_l, p_w):
        s_w = w[:w_l, :]
        ph_s = torch.repeat_interleave(s_w, p_w, dim=0)
        batch.append(ph_s)
    return pad(batch, p_l).to(device)
