# New sample-compression bounds for Pick-To-Learn

This is the repository associated to the paper : 

**Sample compression unleashed : New generalization bounds for real valued losses**

The PBB and regression forests experiments were run using Python 3.12.2 and a virtual environment as defined in the following file :  `requirements_pbb.txt`. All the experiments on MNIST were run using Python 3.12.3 and a virtual environment as defined in the following file :  `requirements_p2l.txt`.

To run the grid search, use the file `main.py`. For the baselines, use the files `baseline.py` and `tree_baseline.py`. The results were parsed using the jupyter notebooks. In both cases, the configs for the datasets can be found in `configs/parameter_configs` and the configs for the type of models can be found in `configs/sweep_configs`.

If you use our code, please cite our paper : 

```
Bazinet, M., Zantedeschi, V., & Germain, P. (2024). Sample compression unleashed: New generalization bounds for real valued losses. arXiv preprint arXiv:2409.17932.
```

```
@article{bazinet2024sample,
  title={Sample compression unleashed: New generalization bounds for real valued losses},
  author={Bazinet, Mathieu and Zantedeschi, Valentina and Germain, Pascal},
  journal={arXiv preprint arXiv:2409.17932},
  year={2024}
}
```


