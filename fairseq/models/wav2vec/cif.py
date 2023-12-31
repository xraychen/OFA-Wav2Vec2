"""
    CIF Function
    https://github.com/George0828Zhang/torch_cif
"""

import torch
from typing import Optional, Tuple
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence


def prob_check(tensor, eps=1e-10, neg_inf=-1e8, logp=False):
    assert not torch.isnan(tensor).any(), (
        "Nan in a probability tensor."
    )
    # Add the eps here to prevent errors introduced by precision
    if logp:
        assert tensor.le(0).all() and tensor.ge(neg_inf).all(), (
            "Incorrect values in a log-probability tensor"
            ", -inf <= tensor <= 0"
        )
    else:
        assert tensor.le(1.0 + eps).all() and tensor.ge(0.0 - eps).all(), (
            "Incorrect values in a probability tensor"
            ", 0.0 <= tensor <= 1.0"
        )


def get_reverse_alpha(alpha, fire_index, origin_length, eps=1e-7):
    csum = alpha.cumsum(-1)
    reverse_alpha = []

    for i, (ind, alpha_i, csum_i) in enumerate(zip(fire_index, alpha, csum)):
        ind = ind.nonzero(as_tuple=True)[0]
        ind_diff = torch.diff(ind)
        ind_diff -= 1
        ind_diff = torch.cat(((ind[0]).unsqueeze(-1), ind_diff)).float()
        right_extra = (csum_i[ind] % 1) / (alpha_i[ind] + eps)
        left_extra =  torch.ones_like(ind) - right_extra
        ind_diff[1:] += right_extra[:-1]
        ind_diff += left_extra
        # extend the last postition if the length is not long enough
        # if ind_diff.sum(-1) < origin_length:
        #     ind_diff[-1] += origin_length - ind_diff.sum(-1)
        reverse_alpha.append(ind_diff)

    reverse_alpha = pad_sequence(reverse_alpha, batch_first=True)
    return reverse_alpha


def cif_function(
    input: Tensor,
    alpha: Tensor,
    beta: float = 1.0,
    gamma: float = 0.0,
    tail_thres: float = 0.5,
    padding_mask: Optional[Tensor] = None,
    target_lengths: Optional[Tensor] = None,
    eps: float = 1e-4,
    cal_reverse_alpha = False,
    is_training = True,
    protect_alpha = True,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    r""" A fast parallel implementation of continuous integrate-and-fire (CIF)
    https://arxiv.org/abs/1905.11235
    Args:
        input (Tensor): (N, S, C) Input features to be integrated.
        alpha (Tensor): (N, S) Weights corresponding to each elements in the
            input. It is expected to be after sigmoid function.
        beta (float): the threshold used for determine firing.
        gamma (float): the ratio to use self alpha, (1 - gamma) * target_len + gamma * predicted_alpha.
        tail_thres (float): the threshold for determine firing for tail handling.
        padding_mask (Tensor, optional): (N, S) A binary mask representing
            padded elements in the input.
        target_lengths (Tensor, optional): (N,) Desired length of the targets
            for each sample in the minibatch.
        eps (float, optional): Epsilon to prevent underflow for divisions.
            Default: 1e-4
    Returns -> Dict[str, List[Optional[Tensor]]]: Key/values described below.
        cif_out (Tensor): (N, T, C) The output integrated from the source.
        cif_lengths (Tensor): (N,) The output length for each element in batch.
        alpha_sum (Tensor): (N,) The sum of alpha for each element in batch.
            Can be used to compute the quantity loss.
        delays (Tensor): (N, T) The expected delay (in terms of source tokens) for
            each target tokens in the batch.
        tail_weights (Tensor, optional): (N,) During inference, return the tail.
    """
    B, S, C = input.size()
    assert tuple(alpha.size()) == (B, S), f"alpha size mismatch: {alpha.size()} != {(B, S)}"
    # prob_check(alpha)

    dtype = alpha.dtype
    alpha = alpha.float()
    if padding_mask is not None:
        padding_mask = padding_mask.bool()
        alpha = alpha.masked_fill(padding_mask, 0)

    if target_lengths is not None:
        feat_lengths = target_lengths.long()
        desired_sum = beta * target_lengths.type_as(input) + eps
        alpha_sum = alpha.sum(dim=1)
        alpha_std = alpha.std(dim=1)
        if gamma != 0.0:
            assert 0.0 < gamma <= 1.0
            desired_sum = (1.0 - gamma) * desired_sum + gamma * alpha_sum
            feat_lengths = (desired_sum / beta).floor().long()

        alpha = alpha * (desired_sum / alpha_sum).unsqueeze(1)
        T = feat_lengths.max()
    else:
        alpha_sum = alpha.sum(dim=1)
        alpha_std = alpha.std(dim=1)

        if protect_alpha:
            # minimum fire num (negative sample needs sample num > 1)
            min_fire_num = 2

            temp = alpha_sum < min_fire_num # negative sameple needs sample num > 1
            if True in temp:
                desired_sum = alpha_sum
                desired_sum[temp] = min_fire_num + eps # negative sameple needs sample num > 1
                alpha = alpha * (desired_sum / alpha_sum).unsqueeze(1)

                alpha_sum = alpha.sum(dim=1)
                alpha_std = alpha.std(dim=1)

        feat_lengths = (alpha_sum / beta).floor().long()
        T = feat_lengths.max()

    # aggregate and integrate
    csum = alpha.cumsum(-1)
    with torch.no_grad():
        # indices used for scattering
        right_idx = (csum / beta).floor().long().clip(max=T)
        left_idx = right_idx.roll(1, dims=1)
        left_idx[:, 0] = 0

        # count # of fires from each source
        fire_num = right_idx - left_idx
        extra_weights = (fire_num - 1).clip(min=0)

    # The extra entry in last dim is for
    output = input.new_zeros((B, T + 1, C))
    delay = input.new_zeros((B, T + 1))
    source_range = torch.arange(1, 1 + S).unsqueeze(0).type_as(input)
    zero = alpha.new_zeros((1,))

    # right scatter
    fire_mask = fire_num > 0
    right_weight = torch.where(
        fire_mask,
        csum - right_idx.type_as(alpha) * beta,
        zero
    ).type_as(input)
    # assert right_weight.ge(0).all(), f"{right_weight} should be non-negative."
    output.scatter_add_(
        1,
        right_idx.unsqueeze(-1).expand(-1, -1, C),
        right_weight.unsqueeze(-1) * input
    )
    delay.scatter_add_(
        1,
        right_idx,
        right_weight * source_range / beta
    )

    # left scatter
    left_weight = (
        alpha - right_weight - extra_weights.type_as(alpha) * beta
    ).type_as(input)
    output.scatter_add_(
        1,
        left_idx.unsqueeze(-1).expand(-1, -1, C),
        left_weight.unsqueeze(-1) * input
    )
    delay.scatter_add_(
        1,
        left_idx,
        left_weight * source_range / beta
    )

    # extra scatters
    if extra_weights.ge(0).any():
        extra_steps = extra_weights.max().item()
        tgt_idx = left_idx
        src_feats = input * beta
        for _ in range(extra_steps):
            tgt_idx = (tgt_idx + 1).clip(max=T)
            # (B, S, 1)
            src_mask = (extra_weights > 0)
            output.scatter_add_(
                1,
                tgt_idx.unsqueeze(-1).expand(-1, -1, C),
                src_feats * src_mask.unsqueeze(2)
            )
            delay.scatter_add_(
                1,
                tgt_idx,
                source_range * src_mask
            )
            extra_weights -= 1

    # tail handling
    if target_lengths is not None or is_training:
        # training time -> ignore tail
        output = output[:, :T, :]
        delay = delay[:, :T]
    else:
        # print('tail handling')
        # find out contribution to output tail
        # note: w/o scaling, extra weight is all 0
        zero = right_weight.new_zeros((1,))
        r_mask = right_idx == feat_lengths.unsqueeze(1)
        tail_weights = torch.where(r_mask, right_weight, zero).sum(-1)
        l_mask = left_idx == feat_lengths.unsqueeze(1)
        tail_weights += torch.where(l_mask, left_weight, zero).sum(-1)

        # a size (B,) mask that extends position that passed threshold.
        extend_mask = tail_weights >= tail_thres

        # extend 1 fire and upscale the weights
        # TODO: fix this block of code which will breaks auto grad
        if extend_mask.any():
            # (B, T, C), may have infs so need the masks
            upscale = (
                torch.ones_like(output).scatter(
                    1,
                    feat_lengths.view(B, 1, 1).expand(-1, -1, C),
                    beta / tail_weights.view(B, 1, 1).expand(-1, -1, C),
                )
            )
            output[extend_mask] *= upscale[extend_mask]
            feat_lengths += extend_mask.long()
            T = feat_lengths.max()
        output = output[:, :T, :]
        delay = delay[:, :T]
        # a size (B, T) mask to erase weights
        tail_mask = torch.arange(T, device=output.device).unsqueeze(0) >= feat_lengths.unsqueeze(1)
        output[tail_mask] = 0

    if cal_reverse_alpha:
        reverse_alpha = get_reverse_alpha(alpha, fire_mask.long(), S)
    else:
        reverse_alpha = None


    return {
        "cif_out": [output],
        "cif_lengths": [feat_lengths],
        "alpha": [alpha],
        "alpha_sum": [alpha_sum.to(dtype)],
        "alpha_std": [alpha_std],
        "delays": [delay],
        "tail_weights": [tail_weights] if (target_lengths is None and not is_training) else [],
        "fire_index": [fire_mask.long()],
        "reverse_alpha": [reverse_alpha] if cal_reverse_alpha else []
    }
