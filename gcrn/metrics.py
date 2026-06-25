import torch


def masked_mae(prediction, target):
    mask = (target != 0).float()
    mask = mask / mask.mean().clamp_min(1e-6)
    error = torch.abs(prediction - target)
    return (error * mask).mean()


def calculate_metrics(prediction, target):
    mask = target != 0
    prediction = prediction[mask]
    target = target[mask]
    mae = torch.mean(torch.abs(prediction - target))
    rmse = torch.sqrt(torch.mean((prediction - target) ** 2))
    percentage_error = torch.abs(
        (prediction - target) / target.clamp_min(1e-5)
    )
    mape = percentage_error.mean() * 100
    return mae.item(), rmse.item(), mape.item()
