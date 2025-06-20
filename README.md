# Cross-Species Connectome Consistency via Inter-Bundle Topological Graph Network (ibTGN)
This project is the source code of a paper published at the __IEEE ISBI 2025__ conference, titled: __Exploring Consistency of Connectome Across Species Using Inter-Bundle Topological Graph Network (ibTGN)__.

## Introduction

Understanding how white matter structures are preserved across species is a fundamental question in comparative neuroscience. However, traditional registration-based methods suffer from inaccuracies due to differences in brain morphology and size, particularly when comparing humans and non-human primates.
It encodes the topological structure of brain tractography as a graph and leverages a (variational) graph autoencoder to learn low-dimensional embeddings that preserve intra-species and inter-species bundle topology. Cross-species homologous bundles are identified in latent space, enabling robust and efficient structural alignment across species.

---

## Key Contributions

- **Registration-Free**: Avoids error-prone cross-species registration.
- **Graph-Based Representation**: Constructs inter-bundle topological graphs from tractography.
- **Homology Discovery**: Identifies morphologically and functionally similar bundles across species.
- **VGAE-based Alignment**: Learns species-agnostic latent embeddings of fiber bundle graphs.

---

## Project Structure

```bash
├── args.py                 # Training configuration
├── cmp_HP.py               # Cross-species comparison and visualization
├── dataset.py              # GraphDataset class for loading input graphs
├── model.py                # GAE/VGAE model definitions and GCN layers
├── test_args.py            # Evaluation/test configuration
├── train.py                # Training script for GAE or VGAE
└── utils
    ├── calc_code_euc.py    # Latent space nearest neighbor search
    ├── input_data.py       # input data code file
    ├── preprocessing.py    # Graph preprocessing utilities
    ├── readTck_Trk.py      # read tck and trk file
    └── readVTK.py          # read vtk file
```

---

## Dependence
```
Python == 3.8
Pytorch == 1.12.1
sklearn == 0.22
networkx == 3.1
dipy == 1.7.0
scipy == 1.8.0
nibabel == 5.2.1
```

---

## Usage

### Step 1: Prepare Data

Place processed fiber clustering `.npy` files in `cmp_data_v2/sub-XXX/cluser_*.npy`.
You can generate these using DSI Studio or pre-supplied data format with `m=15` resampled points per streamline.

### Step 2: Train the Model

Train the model using `GAE` or `VGAE`:

```bash
python train.py
```

Configuration is controlled via `args.py`, e.g.:

```python
model = 'GAE'           # or 'VGAE'
input_dim = 45
hidden1_dim = 32
hidden2_dim = 16
num_epoch = 30
```
Training outputs:
- `trained_model/GAE_model.pth`
- Training curves in `figure/GAE_metric.png`

### Step 3: Cross-Species Evaluation
Run `cmp_HP.py` to:

- Load latent codes
- Search nearest bundles in target species
- Render homologous fibers for comparison

```bash
python cmp_HP.py
```

Modify `test_args.py` to switch direction of comparison (human → monkey or monkey → human) and target subjects.

## Citation
If you find this code useful, please consider citing:

```bibtex
@inproceedings{ibTGN,
  title={Exploring Consistency of Connectome Across Species Using Inter-Bundle Topological Graph Network},
  author={Yazhe Zhai, Yu Xie, Yifei He,	Xinyuan Zhang, Jiaolong Qin, Ye Wu},
  booktitle={International Symposium on Biomedical Imaging 2025(ISBI)},
  year={2025}
}
```
