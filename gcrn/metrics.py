import torch


def masked_mae(prediction, target):
    mask = (target != 0).float()
    mask = mask / mask.mean().clamp_min(1e-6)
    return (torch.abs(prediction - target) * mask).mean()


def calculate_metrics(prediction, target):
    mask = target != 0
    prediction = prediction[mask]
    target = target[mask]
    mae = torch.mean(torch.abs(prediction - target))
    rmse = torch.sqrt(torch.mean((prediction - target) ** 2))
    mape = (
        torch.mean(torch.abs((prediction - target) / target.clamp_min(1e-5)))
        * 100
    )
    return mae.item(), rmse.item(), mape.item()
