[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>

<p align="left">
  <a href="https://github.com/anonymousauthors/gauche">
    <img src="https://raw.githubusercontent.com/leojklarner/gauche/main/imgs/gauche_logo.png" width="45%" />
    <img src="https://github.com/leojklarner/gauche/blob/main/imgs/gauche.gif?raw=true" width="22%" hspace="30"/>
  </a>
</p>

A Gaussian Process Library for Molecules, Proteins and Reactions.

## Install

We recommend using a conda virtual environment:.
```
conda env create -f conda_env.yml

pip install --no-deps rxnfp
pip install --no-deps drfp
pip install transformers
```

Optional for running tests.
```
pip install gpflow grakel
```

## Citing

If GAUCHE is useful for your work please consider citing the following paper:

```
@inproceedings{griffiths2022gauche,
  title={GAUCHE: A Library for Gaussian Processes in Chemistry},
  author={Griffiths, Ryan-Rhys and Klarner, Leo and Moss, Henry and Ravuri, Aditya and Truong, Sang T and Rankovic, Bojana and Du, Yuanqi and Jamasb, Arian Rokkum and Schwartz, Julius and Tripp, Austin and others},
  booktitle={ICML 2022 2nd AI for Science Workshop}
}

```
