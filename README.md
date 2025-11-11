[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![license](https://img.shields.io/badge/license-apache_2.0-orange.svg)](https://opensource.org/licenses/Apache-2.0)


## $K$ -adaptability for two-stage stochastic optimization

Implementation of the PiM methdology proposed in 

[K-adaptability for two-stage stochastic optimization. (2025)]

### Main Dependencies Installation

In order to execute the code, you need an [Anaconda](https://www.anaconda.com/) environment with Python>=3.10.

For the packages installation, open a terminal (Anaconda Prompt for Windows users) in the project root folder and execute the following commands.

```
pip install tabulate
pip install psutil
pip install gurobipy
pip install numpy
pip install sympy
pip install joblib
```

### Usage

- "kp_OR" stands for stochastic knapsack 
- "tskp" stands for two-stage stochastic knapsack problem with setup. "tskp_OR" contains the original formulation of the problem, while in "tskp" additional constraints requiring a minimum capacity be filled are considered.
- "qtskp" stands for quadratic two-stage stochastic knapsack problem with setup. Additional constraints of minimum capacity are considered.
- "cfl" stands for two-stage stochastic capacitated facility location problem. "cfl_OR" contains the original formulation of the problem, while in "cfl" additional constraints requiring a minimum capacity be filled are considered.
- "qcfl" stands for quadratic two-stage capacitated facility location problem. Additional constraints of minimum capacity are considered.

Files starting with "pf" contain problem specific functions: the subpartition induced problem is defined for the corresponding underlying problem in the function ``` spip_build() ```.

In ``` solver.py ``` implementations of PiM, CE and MIQP can be found.

In ``` environment.py ``` the data required for each class of problems is defined.

Different values of $K$, $\ell$ or $n_x,n_y$ can be specified in the files starting with "main".

Given a terminal (Anaconda Prompt for Windows users), an example of execution could be the following.

``` python main_tskp.py ```
