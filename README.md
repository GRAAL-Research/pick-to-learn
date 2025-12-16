# New sample-compression bounds for Pick-To-Learn

This is the repository associated to the paper : 

**Sample compression unleashed : New generalization bounds for real valued losses**

The PBB and regression forests experiments were run using Python 3.12.2 and a virtual environment as defined in the following file :  `requirements_pbb.txt`. All the experiments on MNIST were run using Python 3.12.3 and a virtual environment as defined in the following file :  `requirements_p2l.txt`.

To run the grid search, use the file `main.py`. For the baselines, use the files `baseline.py` and `tree_baseline.py`. The results were parsed using the jupyter notebooks. In both cases, the configs for the datasets can be found in `configs/parameter_configs` and the configs for the type of models can be found in `configs/sweep_configs`.

If you use our code, please cite our paper : 

```
Bazinet, M., Zantedeschi, V., & Germain, P. (2025). Sample compression unleashed: New generalization bounds for real valued losses. In The 28th International Conference on Artificial Intelligence and Statistics.
```

```
@inproceedings{
bazinet2025sample,
title={Sample compression unleashed: New generalization bounds for real valued losses},
author={Mathieu Bazinet and Valentina Zantedeschi and Pascal Germain},
booktitle={The 28th International Conference on Artificial Intelligence and Statistics},
year={2025},
url={https://openreview.net/forum?id=0ynSy2dwNi}
}
```

---- 
#### Erratum on December 16th, 2025

There was a small mistake in the code for the Pick-To-Learn bound, which led to slightly tighter bounds than expected in the paper. For future use, this problem is now resolved. 