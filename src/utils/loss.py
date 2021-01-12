import torch

joint_id_to_name = {
  0: 'Head',        8: 'Torso',
  1: 'Neck',        9: 'R Hip',
  2: 'R Shoulder',  10: 'L Hip',
  3: 'L Shoulder',  11: 'R Knee',
  4: 'R Elbow',     12: 'L Knee',
  5: 'L Elbow',     13: 'R Foot',
  6: 'R Hand',      14: 'L Foot',
  7: 'L Hand',
}

def mse_masked(input, target, mask):
    """Mean squared error with visibility mask implementation
    Note:
        If mask is all zero => loss should be zero
    """
    return torch.mean(torch.sum((input - target) ** 2, dim=2) * (mask == 1))

def _vitruvian_calculate(input, index, mask):
    """Calculate distance on given index, masked for visibility"""
    m, _ = torch.min((mask[:, index] != 0), dim=1)
    dist_sx = torch.sqrt(torch.sum((input[:, index[0]] - input[:, index[1]]) ** 2, dim=1) + 1e-9)[m]
    dist_dx = torch.sqrt(torch.sum((input[:, index[2]] - input[:, index[3]]) ** 2, dim=1) + 1e-9)[m]
    if dist_sx.size(0) == 0 or dist_dx.size(0) == 0:
        return 0.
    else:
        return torch.mean(torch.abs(dist_sx - dist_dx))

def vitruvian_loss(input, mask, dataset):
    """Vitruvian loss implementation"""
    if dataset == "itop":
        # 1 - 2 e 1 - 3 -> collo spalle
        # 2 - 4 e 3 - 5 -> spalle gomito
        # 4 - 6 e 5 - 7 -> gomito mano
        # 9 - 11 e 10 - 12 -> anca ginocchio
        # 11 - 13 e 12 - 14 -> ginocchio piede
        loss = _vitruvian_calculate(input, [1, 2, 1, 3], mask)
        loss += _vitruvian_calculate(input, [2, 4, 3, 5], mask)
        loss += _vitruvian_calculate(input, [4, 6, 5, 7], mask)
        loss += _vitruvian_calculate(input, [9, 11, 10, 12], mask)
        loss += _vitruvian_calculate(input, [11, 13, 12, 14], mask)
    elif dataset in ("watch_n_patch", "wnp", "watch-n-patch"):
        # 20 - 4 e 20 - 8 -> spine shoulder spalle
        # 4 - 5 e 8 - 9 -> spalle gomito
        # 5 - 6 e 9 - 10 -> gomito polso
        # 6 - 7 e 10 - 11 -> polso mano
        # 12 - 0 e 0 - 16 -> anche spine base
        # 12 - 13 e 16 - 17 -> anca ginocchio
        # 13 - 14 e 17 - 18 -> ginocchio caviglia
        # 14 - 15 e 18 - 19 -> caviglia piede
        limbs = [
            [20, 4, 20, 8],
            [4, 5, 8, 9],
            [5, 6, 9, 10],
            [6, 7, 10, 11],
            [0, 12, 0, 16],
            [12, 13, 16, 17],
            [13, 14, 17, 18],
            [14, 15, 18, 19],
        ]
        loss = 0.0
        for limb in limbs:
            loss += _vitruvian_calculate(input, limb, mask)
    return loss


def forcing_loss(input, target, mask, dataset):
    """Forcing loss implementation"""
    if dataset == "itop":
        feet = [13, 14]
        head = 0
        torso = 8
    elif dataset in ("watch_n_patch", "wnp", "watch-n-patch"):
        feet = [15, 19]
        head = 3
        torso = 1
    else:
        raise ValueError
    m = (((mask[:, feet[0]] != 0) | (mask[:, feet[1]] != 0)) & (mask[:, head]  != 0)).type(torch.cuda.FloatTensor)

    height = (target[:, head, 1] - torch.min(target[:, [feet[0], feet[1]], 1])) * m

    min = target[:, torso, 2] - (height / 1.5)
    max = target[:, torso, 2] + (height / 1.5)
    mid = target[:, torso, 2].unsqueeze(-1).repeat(1, input.shape[1])
    min = min.unsqueeze(-1).repeat(1, input.shape[1])
    max = max.unsqueeze(-1).repeat(1, input.shape[1])

    masked = ((input[:, :, 2] < min) | (input[:, :, 2] > max)).type(torch.cuda.FloatTensor) * mask * m.unsqueeze(-1)
    loss = torch.mean(torch.sum(torch.abs((input[:, :, 2] - mid) * masked), dim=1))

    return loss
