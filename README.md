# AI molecular dynamics simulation for bio-chemical targets

本仓库实现了基用于生物大分子于深度学习方法的分子动力学方法,并给出完整的应用实例，在前人的基础上，给出了示例数据、数据预处理代码、以及安装流程，以便后续研究学者能够复现文章的内容以及改进现有模型。

## Installation

运行本仓库代码，需要安装gromacs软件以处理分子动力学模拟数据。本仓库代码基于pytorch框架编写。为了运行本仓库代码，你需要准备以下系统环境以及Python环境

### Requirements

<ul>
<li>ubuntu 20.04</li>
<li>gcc 8.4.0</li>
<li>gromacs 2022.4 </li>
</ul>

### Library

To install latex package

```
sudo apt install dvipng texlive texlive-latex-extra texlive-latex-recommended cm-super
```

### Python

```
conda env create -f led.yaml
pip install -r requirements.txt
```

## Data Preprocess

我们用OpenMM软件模拟了几个蛋白质的运动轨迹，并提供了相应的数据预处理代码。根据分子的topol文件，计算出分子运动轨迹每一帧的特征，如键长、键角以及二面角等（忽略氢原子）。当你想应用到新的分子时，你需要提供平衡后的xtc轨迹文件以及模拟时的topol文件，并对筛选的原子类型稍作修改即可。

## RUNNING

为了验证LED模型在

### Train

### Test

## Analysis


## References
