# ibTGN
### EXPLORING CONSISTENCY OF CONNECTOME ACROSS SPECIES USING INTER-BUNDLE TOPOLOGICAL GRAPH NETWORK (IBTGN)

This project is the source code of a paper published at the __IEEE ISBI 2025__ conference, titled: __Exploring Consistency of Connectome Across Species Using Inter-Bundle Topological Graph Network (ibTGN)__.

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
