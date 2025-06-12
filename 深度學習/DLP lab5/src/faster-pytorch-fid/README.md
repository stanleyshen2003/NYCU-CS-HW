# Faster FID score for PyTorch

This repository provides a modified implementation of [pytorch-fid](https://github.com/mseitzer/pytorch-fid) that accelerates the computation of the Fréchet Inception Distance (FID) score by replacing the computation of the matrix square root, originally implemented with SciPy on CPU, with PyTorch on GPU. This implementation can compute the FID score (`calculate_frechet_distance` function) up to **10 times faster**, reducing the computation time from 6 minutes to 30 seconds on a GPU for a feature dimensionality of 2048.

## Environment Setting

To use this implementation, clone the repository and navigate to its directory:
```
git clone https://github.com/jaywu109/faster-pytorch-fid.git
cd faster-pytorch-fid
```
This implementation requires the following dependencies:
- python3
- pytorch
- torchvision
- pillow
- numpy
- scipy

The implementation has been tested on the following package versions:
- python==3.9.16 
- pytorch==1.12.1
- torchvision==0.13.1
- pillow==9.4.0 
- numpy==1.22.3  
- scipy==1.10.1 

## Usage

To compute the FID score between two datasets, where images of each dataset are contained in an individual folder:
```
python -m fid_score_gpu.py path/to/dataset1 path/to/dataset2
```

To run the evaluation on GPU, use the flag `--device cuda:N`, where `N` is the index of the GPU to use. **Note that this implementation would not show significant improvement if run on a CPU.**

## Detailed Refinement

The original implementation of pytorch-fid often takes a long time to compute the FID score. After investigating the implementation, we identified that the major bottleneck is the computation of the matrix square root with SciPy's [scipy.linalg.sqrtm](https://github.com/scipy/scipy/blob/v1.10.1/scipy/linalg/_matfuncs_sqrtm.py#L117-L210) function, which involves multiple dot product operations with NumPy. For feature dimensionality of 2048 (final average pooling features), it can take a long time to compute dot products with CPU for matrices of size 2048x2048.

We attempted to use CuPy to replace the SciPy's sqrtm function, but found that it is currently [not supported](https://docs.cupy.dev/en/stable/reference/comparison.html). Instead, we focused on optimizing the most time-consuming part of the function by replacing the dot product operation with `torch.matmul` and running it on a GPU. This approach was simpler than changing all NumPy operations in sqrtm to their PyTorch equivalents, which may not have resulted in significant improvement. An example of the modified code can be seen in the `torch_sqrtm.py` file.

https://github.com/jaywu109/faster-pytorch-fid/blob/e2073cbc99c2d4840d60b5654bd0ed0decadb670/torch_sqrtm.py#L40-L43


## Citing

If you use this repository in your research, consider citing it using the following Bibtex entry:

```
@misc{githubGitHubJaywu109fasterpytorchfid,
  author = {Wu, Dai-Jie},
  title = {{G}it{H}ub - jaywu109/faster-pytorch-fid --- github.com},
  howpublished = {\url{https://github.com/jaywu109/faster-pytorch-fid}},
  year = {2023},
}

@misc{Seitzer2020FID,
  author={Maximilian Seitzer},
  title={{pytorch-fid: FID Score for PyTorch}},
  month={August},
  year={2020},
  note={Version 0.3.0},
  howpublished={\url{https://github.com/mseitzer/pytorch-fid}},
}
```

## License

This implementation is licensed under the Apache License 2.0.

FID was introduced by Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler and Sepp Hochreiter in "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium", see [https://arxiv.org/abs/1706.08500](https://arxiv.org/abs/1706.08500)

The original implementation is by the Institute of Bioinformatics, JKU Linz, licensed under the Apache License 2.0.
See [https://github.com/bioinf-jku/TTUR](https://github.com/bioinf-jku/TTUR).

This is the refined version of **pytorch-fid**. See [https://github.com/mseitzer/pytorch-fid](https://github.com/mseitzer/pytorch-fid).
