# Cross-Species Connectome Consistency via Inter-Bundle Topological Graph Network (ibTGN)
This project is the source code of a paper published at the __IEEE ISBI 2025__ conference, titled: __Exploring Consistency of Connectome Across Species Using Inter-Bundle Topological Graph Network (ibTGN)__.

## Introduction

Understanding how white matter structures are preserved across species is a fundamental question in comparative neuroscience. However, traditional registration-based methods suffer from inaccuracies due to differences in brain morphology and size, particularly when comparing humans and non-human primates.
It encodes the topological structure of brain tractography as a graph and leverages a (variational) graph autoencoder to learn low-dimensional embeddings that preserve intra-species and inter-species bundle topology. Cross-species homologous bundles are identified in latent space, enabling robust and efficient structural alignment across species.

---

# Key Contributions

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


### Dependence
```
Python == 3.8
Pytorch == 1.12.1
sklearn == 0.22
networkx == 3.1
dipy == 1.7.0
scipy == 1.8.0
nibabel == 5.2.1
```

### Usage
You can change the file "./utils/input_data.py" code to your construction graph data reading code. And you must change "args.py" for training metrics.

Then you can run
```
python train.py
```
It saves the training stage metrics in figure. And the weight file saves in "trained_model".

Next, use the weight model and run compare scripts. 
```
python cmp_HP.py
```

At the same time, you should modify the file: "test_args.py" to set the metics. 
