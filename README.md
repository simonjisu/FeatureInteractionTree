# Feature Interaction Tree

**Demo**: [https://simonjisu-featureinteractiontree-app-8shmvb.streamlitapp.com](https://simonjisu-featureinteractiontree-app-8shmvb.streamlitapp.com/) 

Graphviz program is needed.

For Windows/Mac User:

- install graphviz(2.49.3) version from https://graphviz.org/download/

For Ubuntu User:

```
$ sudo apt-get install graphviz libgraphviz-dev pkg-config
```

# Package Install

## For Linux/MacOS Users

For demo purpose user please install package using `pipenv`

```
$ pipenv install
```

For pip/conda user

```
$ pip install -r requirements.txt
```

## For Window Users

```
$ pip install -r requirements_win.txt
```

Install `pygraphviz` 

```
pip install --global-option=build_ext --global-option="-IC:/Program Files/Graphviz/include" --global-option="-LC:/Program Files/Graphviz/lib" pygraphviz
```

# Requirements

```
pygraphviz
anytree
xgboost
ipykernel
scikit-learn
shap
```
