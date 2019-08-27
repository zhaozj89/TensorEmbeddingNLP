# Lexical Feature Embedding via Tensor Decomposition for Humor Recognition

## Install

* `pip install -r requirements.txt`

* install [Matlab Python engine](https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html)

* `git clone git@gitlab.com:tensors/tensor_toolbox.git`

## Methods

* `load_data.py` loads raw text as a N by D index matrix

* `doc2vec.py` calculates tensor embeddings for each sentence

* `label_progagation.py` calculates labels with label propagation

* `tensor_decomp_twitter.py` calculates lexical centrality

## Reproducible Experiments

####  Global Humor Ranking

* download SemEval 2017 Task dataset, and put it into `./data`

* `python tensor_decomp_twitter.py`

* `python taskb_eval_script.py`

####  Binary Humor Classification

* put `16000 one liners` and `pun of the day` datasets into `./data`

* `python cv_portion.py --option <option> --label_portion <label_portion>`, where `<option>` can be `16000oneliners` or `Pun` (corresponding to the `16000 one liners` and `pun of the day` datasets, respectively), `<label_portion>` is a float number of training percentage, such as 0.1

## Issue Solving

* the dataset paths are configurated in `config.py`

----

GNU GENERAL PUBLIC LICENSE
