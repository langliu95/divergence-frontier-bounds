# divergence-frontier-bounds

This repository contains the code to reproduce the experiments 
in this [paper](https://proceedings.neurips.cc/paper/2021/file/6bf733bb7f81e866306e9b5f012419cb-Supplemental.pdf).
The paper studies the sample complexity of the evaluation framework—divergence frontiers—for generative models.
It also introduces frontier integrals as summary statistics of divergence frontiers.

_**Standalone package**: For a self-contained package to compute divergence frontiers and frontier integrals, installable 
via `pip install mauve-text`, please 
see [this repository](https://github.com/krishnap25/mauve)._

## Dependencies
The code is written in Python and the dependencies are:
- python >= 3.6
- ipykernel >= 6.4.1
- matplotlib >= 3.5.0
- numpy >= 1.19.1
- pandas >= 1.3.4
- pathos >= 0.2.8
- scikit-learn >= 0.22.1
- scipy >= 1.7.3
- faiss-cpu >= 1.7.1

**Conda Environment**:
We recommend using a [conda environment](https://docs.conda.io/en/latest/miniconda.html).
To setup the environment, run
```bash
conda env create --file environment.yml
# activate the environment
conda activate divergence-frontier-bounds
python -m ipykernel install --user --name divergence-frontier-bounds
```

## Files

* `metrics.py`: code to compute divergence frontiers and frontier integrals.
* `utils.py`: utility functions.
* `experiments.ipynb`: Jupyter notebook to run experiments and produce plots.
* `parsed_outputs`: 
* `results/`: experimental results.


## Citation
If you find this repository useful, or you use it in your research, please cite:
```
@inproceedings{liu-etal:divergence:neurips2021,
  title={{Divergence Frontiers for Generative Models: Sample Complexity, Quantization Effects, and Frontier Integrals}},
  author={Liu, Lang and Pillutla, Krishna and Welleck, Sean and Oh, Sewoong and Choi, Yejin and Harchaoui, Zaid},
  booktitle={NeurIPS},
  year={2021}
}
```
    
## Acknowledgements
This work was supported by NSF DMS-2134012, NSF CCF-2019844, the CIFAR program “Learning
in Machines and Brains”, and faculty research awards.
