# Finer Granularity Explanation

Graphviz program is needed.

For Windows/Mac User:

- install graphviz(2.49.3) version from https://graphviz.org/download/

For Ubuntu User:

```
$ sudo apt-get install graphviz libgraphviz-dev pkg-config
```

# Package Install

For demo purpose user please install package using `pipenv`

```
$ pipenv install
```

For pip/conda user

```
$ pip install -r requirements.txt
```

Install `pygraphviz` on Windows Users

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
