import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml

from gcrn.data import load_dataset
from gcrn.metrics import calculate_metrics, masked_mae
from gcrn.model import GCRN


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, scaler, adj, device, max_batches=None):
    model.eval()
    predictions, targets = [], []
    for batch_idx, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        output = scaler.inverse_transform(model(x, adj)).cpu()
        predictions.append(output)
        targets.append(scaler.inverse_transform(y).cpu())
        if max_batches is not None and batch_idx + 1 >= max_batches:
            break
    return calculate_metrics(torch.cat(predictions), torch.cat(targets))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pems04.yaml")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.smoke_test:
        config["epochs"] = 1
        config["hidden_dim"] = min(config["hidden_dim"], 16)

    seed_everything(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders, scaler, adj = load_dataset(config)
    adj = adj.to(device)
    model = GCRN(
        input_dim=1,
        hidden_dim=config["hidden_dim"],
        output_steps=config["output_steps"],
        num_layers=config["num_layers"],
        cheb_order=config["cheb_order"],
        dropout=config["dropout"],
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    output_dir = Path("outputs") / config["dataset"]
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best.pt"
    best_val = float("inf")
    epochs_without_improvement = 0
    num_params = sum(p.numel() for p in model.parameters())
    print(f"device={device} parameters={num_params:,}")

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        losses = []
        for step, (x, y) in enumerate(loaders["train"]):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = masked_mae(model(x, adj), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(loss.item())
            if args.smoke_test and step >= 2:
                break

        val = evaluate(
            model,
            loaders["val"],
            scaler,
            adj,
            device,
            max_batches=3 if args.smoke_test else None,
        )
        print(
            f"epoch={epoch:03d} train_loss={np.mean(losses):.4f} "
            f"val_MAE={val[0]:.4f} val_RMSE={val[1]:.4f} "
            f"val_MAPE={val[2]:.2f}%"
        )
        if val[0] < best_val:
            best_val = val[0]
            epochs_without_improvement = 0
            torch.save({"model": model.state_dict(), "config": config}, best_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config["patience"]:
                print("early stopping")
                break

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    test = evaluate(
        model,
        loaders["test"],
        scaler,
        adj,
        device,
        max_batches=3 if args.smoke_test else None,
    )
    result = {"MAE": test[0], "RMSE": test[1], "MAPE": test[2]}
    (output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print("test:", json.dumps(result))


if __name__ == "__main__":
    main()
