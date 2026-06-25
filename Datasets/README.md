# Datasets

This project supports the PEMS04 and PEMS08 traffic-flow datasets. Dataset
files are not included in this repository. Please download them from their
original or authorized public source and place them in the structure shown
below.

## Directory structure

```text
Datasets/
├── PEMS04/
│   ├── PEMS04.npz
│   └── PEMS04.csv
└── PEMS08/
    ├── PEMS08.npz
    └── PEMS08.csv
```

File names are case-sensitive on Linux.


## Data split

Samples are split chronologically:

- 60% training
- 20% validation
- 20% testing

The default setting uses 12 historical time steps to predict the following 12
time steps. These values can be changed in `configs/pems04.yaml` and
`configs/pems08.yaml`.


## License and attribution

PEMS04 and PEMS08 are third-party datasets and are not covered by this
repository's software license. Before redistributing the data, review the
terms of the source from which you obtained it. When publishing results,
please cite the original dataset source and the relevant research papers.
