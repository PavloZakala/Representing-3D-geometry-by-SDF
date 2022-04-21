import torch
import torch.nn.functional as F


def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad


def siren_loss(losses):
    train_loss = 0.
    for loss_name, loss in losses.items():
        train_loss += loss.mean()

    return train_loss


def sdf(model_output, gt):
    '''
       x: batch of input coordinates
       y: usually the output of the trial_soln function
       '''
    gt_sdf = gt['sdf']
    gt_normals = gt['normals']

    coords = model_output['model_in']
    pred_sdf = model_output['model_out']

    grad = gradient(pred_sdf, coords)

    # Wherever boundary_values is not equal to zero, we interpret it as a boundary constraint.
    sdf_constraint = torch.where(gt_sdf != -1, pred_sdf, torch.zeros_like(pred_sdf))
    inter_constraint = torch.where(gt_sdf != -1, torch.zeros_like(pred_sdf), torch.exp(-1e2 * torch.abs(pred_sdf)))
    normal_constraint = torch.where(gt_sdf != -1, 1 - F.cosine_similarity(grad, gt_normals, dim=-1)[..., None],
                                    torch.zeros_like(grad[..., :1]))
    grad_constraint = torch.abs(grad.norm(dim=-1) - 1)
    # Exp      # Lapl
    # -----------------
    return siren_loss({'sdf': torch.abs(sdf_constraint).mean() * 3e3,  # 1e4      # 3e3
                       'inter': inter_constraint.mean() * 1e2,  # 1e2                   # 1e3
                       'normal_constraint': normal_constraint.mean() * 1e2,  # 1e2
                       'grad_constraint': grad_constraint.mean() * 5e1})  # 1e1      # 5e1
