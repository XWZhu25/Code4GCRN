# GCRN Traffic Flow Prediction Reproduction

This is a runnable reproduction of **Urban traffic flow prediction via
data-driven spatiotemporal model**.

Implemented components:

- polynomial spectral graph convolution;
- graph-convolutional gated recurrent units;
- joint spatiotemporal attention;
- PEMS04 and PEMS08 preprocessing;
- masked MAE training and MAE/RMSE/MAPE evaluation;
- early stopping and checkpoint saving.

## Reproducibility assumptions

The paper does not disclose the data split, history/prediction window, hidden
dimensions, graph threshold, or complete training settings. This project uses
common traffic-forecasting settings: chronological 60/20/20 splitting and 12
historical 5-minute steps to predict the next 12 steps. Every assumption is
editable in `configs/`.

The current code implements the predictive GCRN network. IWOA is a separate
hyperparameter-search layer rather than part of the model forward pass. The
paper does not report enough search bounds to reproduce its exact search.

## Run

Quick GPU pipeline check:

```powershell
C:\Users\Xiaowen Zhu\.conda\envs\trace-rag\python.exe train.py --config configs/pems04.yaml --smoke-test
```

Full experiments:

```powershell
C:\Users\Xiaowen Zhu\.conda\envs\trace-rag\python.exe train.py --config configs/pems04.yaml
C:\Users\Xiaowen Zhu\.conda\envs\trace-rag\python.exe train.py --config configs/pems08.yaml
```

Checkpoints and metrics are saved under `outputs/<dataset>/`.
