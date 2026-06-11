from functools import lru_cache

import torch
import torch.distributed as dist


@lru_cache(maxsize=5)
def create_pos_mask(num_views, batch_size, device):
    """
    Create a binary mask for positive pairs for multiple views.

    Each instance has num_views augmented samples.
    Positives are all other views of the same instance (excluding self).
    """
    n = batch_size * num_views
    pos_mask = torch.zeros((n, n), dtype=torch.bool, device=device)

    for i in range(batch_size):
        indices = [i + j * batch_size for j in range(num_views)]
        for m in indices:
            for n in indices:
                if m != n:
                    pos_mask[m, n] = True

    return pos_mask


def create_neg_mask(pos_mask):
    """
    Create a binary mask for negative pairs based on the positive mask.

    Args:
        pos_mask (Tensor): Positive mask.

    Returns:
        Tensor: Negative mask.
    """
    neg_mask = pos_mask.clone()
    neg_mask = neg_mask.fill_diagonal_(True)
    neg_mask = ~neg_mask
    return neg_mask


@lru_cache(maxsize=2)
def get_arange(bs, device):
    batch_idx = torch.arange(bs, device=device)
    return batch_idx


class SinkhornKnopp(torch.nn.Module):

    @torch.no_grad()
    def forward(self, teacher_output, teacher_temp, n_iterations=3):
        teacher_output = teacher_output.float()
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        Q = torch.exp(
            teacher_output / teacher_temp
        ).t()  # Q is K-by-B for consistency with notations from our paper
        B = Q.shape[1] * world_size  # number of samples to assign
        K = Q.shape[0]  # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        if dist.is_initialized():
            dist.all_reduce(sum_Q)
        Q /= sum_Q

        for it in range(n_iterations):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            if dist.is_initialized():
                dist.all_reduce(sum_of_rows)
            Q /= sum_of_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B  # the columns must sum to 1 so that Q is an assignment
        return Q.t()
