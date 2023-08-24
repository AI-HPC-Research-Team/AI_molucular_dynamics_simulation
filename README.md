# AI molecular dynamics simulation for bio-chemical targets

This repository provides complete application examples of [LED](https://github.com/cselab/LED-Molecular.git)(molecular dynamics simulation based on deep learning). Based on [LED](https://github.com/cselab/LED-Molecular.git), it provides sample data, data preprocessing code, and installation procedures.
## Installation
To run the code in this repository, you need to prepare the following system environment and Python environment.
### Requirements

<ul>
<li>Ubuntu 20.04</li>
<li>GCC 8.4.0</li>
<li>GROMACS 2022.4 </li>
</ul>

### Library

To install LaTex package

```sh
sudo apt install dvipng texlive texlive-latex-extra texlive-latex-recommended cm-super zliblg-devel
```

### Python

```sh
conda env create -f led.yaml
pip install -r requirements.txt
```

## Data Preprocess
We simulated the motion trajectories of several proteins using the OpenMM software and provided the corresponding data preprocessing code. Based on the topology file of the molecule, we calculated the features of each frame of the molecular motion trajectory, such as bond length, bond angle, and dihedral angle (ignoring hydrogen atoms). When you want to apply it to new molecules, you need to provide the equilibrated xtc trajectory file and the topology file during the simulation, and make slight modifications.


## Adaptation for new molecules
Taking the CALMODULIN molecule as an example, the following modifications need to be made:
<ol>
<li>Copy the script "Modified_LED-Molecular/Experiments/TRP/MDN-ARNN-Local.sh" for the new molecule and store it in the "Experiment/CALMODULIN"</li>
<li>Modify the MDN-ARNN-Local.sh</li> 
<li>Modify the python file</li>
</ol> 
### MDN-ARNN-Local.sh
Modify the following parameter in file 'MDN-ARNN-Local.sh' according to the new molecule:
```sh
system_name=CALMODULIN
input_dim=3447
MDN_distribution=calmodulin
```
To improve the model performance, you can adjust the following model parameters in 'MDN-ARNN-Local.sh':
```
AE_layers_num
AE_layer_size
MDN_kernerls
MDN_hidden_units
RNN_MDN_kernels
RNN_MDN_hidden_units
```

### Python file
<ol>
<li>Implement 'defineCalmodulinVars' function in file 'mixture_density.py'</li>
<li>Create a new folder under 'Modified_LED-Molecular/Methods/LED/Systems'
<li>Implement 'computeLatentDynamicsDistributionErrorCalmodulin' in new file 'utils_plotting_calmodulin.py' </li>
<li>Implement addResultsSystemCalmodulin' in new file 'utils_processing_calmodulin.py' </li>
<li>Implement corresponding plotting code in file 'system_plotting.py'</li>
<li>Implement corresponding processing code in file 'system_processing.py'</li>
</ol>

## RUNNING
```sh
cd Modified_LED-Molecular/Experiments/YourSystemName
bash MDN-ARNN-Local.sh
```
## References
[1] Vlachas P R, Zavadlav J, Praprotnik M, et al. Accelerated simulations of molecular systems through learning of effective dynamics[J]. Journal of Chemical Theory and Computation, 2021, 18(1): 538-549.
[2] Code: [LED](https://github.com/cselab/LED-Molecular.git)