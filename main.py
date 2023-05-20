print(
"""

# Coarse Graining of Chignolin

In this tutorial, we provide instructions on training and simulating a coarse-grained model of miniprotein - Chignolin. We will go through the whole process in a few steps:

Data preparation
Training
Simulation
Analysis

More details and theoretical background is provided in the corresponding publication.
"""
)

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from moleculekit.molecule import Molecule
from moleculekit.projections.metricrmsd import MetricRmsd
from scipy.optimize import curve_fit
from tqdm import tqdm

sns.set_style("whitegrid")


print(
"""
1. Data preparation

In this section, we will produce all the files necessary for training and simulation.
1.1. PDB to PSF conversion

In this point, we will convert PDB file with full atom Chignolin chain to topology (PSF) file containing a coarse-grained system. For this purpose, we use CLN025 variant of chignolin.

In this example, the coarse-grained system consists only of CA atoms that are connected by "bonds". To build such system we use pdb2psf_CA function. Bonded energy term and repulsion will be the only prior terms we use, therefore we don't need to define angles, dihedrals and impropers in PSF file.

The bead names indicate that it's CA atom and a third letter corresponds to a one-letter abbreviation of the aminoacid, eg. CAA for alanine.

"""
)



from torchmd_cg.utils.psfwriter import pdb2psf_CA

PDB_file = 'data/chignolin_cln025.pdb'
PSF_file = 'data/chignolin_ca_top.psf'

pdb2psf_CA(PDB_file, PSF_file, bonds = True, angles = False)


print(
"""

1.2. Prior force parameters extraction form MD data

In this part, we will extract the parameters for prior forces from MD data.

We will use only the subset of data. However, a forcefield file computed based on full simulation data is provided in data/chignolin_priors_fulldata.yaml

The functions defined below correspond to force terms defined in torchMD simulation package.

    harmonic to bonded term (Bonds)
    CG to repulsive term (RepulsionCG)

Load the data

Data provided here is a subset of full simulation data for chignolin. It is supposed to act as a simple example of data preparation for this tutorial.

    data presented here - 18689 frames

    the original set - 1868861 frames
"""
)

mol = Molecule('data/chignolin_ca_top.psf')
arr = np.load('data/chignolin_ca_coords.npy')


mol.coords = np.moveaxis(arr, 0, -1) 

print(
"""
Initiate Prior dictionary

The prior dictionary, that will form a force field file, need to be filled with fields:

    atomtypes - stores unique bead names
    bonds - parameters describing bonded interactions. Both parameters will be calculated based on the training data.
        req - equilibrium distance of the bond
        k0 - spring constant
    lj - parameters describing Lennard-Jones interactions.
        epsilon - will be calculated based on the training data.
        sigma - in this case set to 1.0

In this example function fitted to data is not a Lennard-Jones potential, but rather a LJ-inspired function described by the equation:

V = 4*eps*((sigma/r)**6) + V0

    electrostatics - parameters describing electrostatic interactions:
        charge - in this case
    masses - masses of the beads. In here 12, because all beads correspond to carbon atoms
"""
)

priors = {}
priors['atomtypes'] = list(set(mol.atomtype))
priors['bonds'] = {}
priors['lj'] = {}
priors['electrostatics'] = {at: {'charge': 0.0} for at in priors['atomtypes']}
priors['masses'] = {at: 12.0 for at in priors['atomtypes']}

print(
"""
1.2.1 Bonded interactions

First bonded interactions approximated by harmonic function:

V = k * (x - x0)**2 + V0

The fit of the function to data should be inspected and if needed the range of histogram adjusted by changing fit_range.

"""
)


from torchmd_cg.utils.prior_fit import get_param_bonded

T = 350 # K
fit_range = [3.55,4.2]

bond_params = get_param_bonded(mol, fit_range, T)

priors['bonds'] = bond_params


print(
'''
1.2.2 Non-bonded interactions

Next non-bonded interactions (interactions between atoms that are not connected by bonds) approximated by a custom function inspired by the repulsive term of Lennard-Jones potential described by the equation:

V = 4 * eps * ((sigma/r)**6) + V0

The fit of the function to data should be inspected and if needed the range of histogram adjusted.
'''
)


from torchmd_cg.utils.prior_fit import get_param_nonbonded

fit_range = [3, 6]

nonbond_params = get_param_nonbonded(mol, fit_range, T)

priors['lj'] = nonbond_params



print(
'''
Now let's write YAML file with forcefield parameters.
'''
)

with open("data/chignolin_priors.yaml","w") as f: 
    yaml.dump(priors, f)


print(
'''
1.3. Delta-forces Preparation

Now we can use the force field file created in the previous section to extract delta forces that will serve as an input for training.

To compute delta forces we use make_deltaforces function
'''
)

from torchmd_cg.utils.make_deltaforces import make_deltaforces

coords_npz = 'data/chignolin_ca_coords.npy'
forces_npz = 'data/chignolin_ca_forces.npy'
delta_forces_npz = 'data/chignolin_ca_deltaforces.npy'      # this is the name of the file to be saved; delta_forces_npz is created by make_deltaforces
forcefield = 'data/chignolin_priors.yaml' 
psf = 'data/chignolin_ca_top.psf'
exclusions = ('bonds')
device = 'cpu' 
# forceterms = ['Bonds','RepulsionCG'],
forceterms = ['bonds','repulsioncg','lj']
make_deltaforces(coords_npz, forces_npz, delta_forces_npz, forcefield, psf, exclusions, device, forceterms)


print(
"""
1.4. Embedding Preparation

Now, let's convert bead names to embeddings, that will be used as an input for training. To each residue assign different number, as listed in a dictionary below.
"""
)

AA2INT = {'ALA':1,
         'GLY':2,
         'PHE':3,
         'TYR':4,
         'ASP':5,
         'GLU':6,
         'TRP':7,
         'PRO':8,
         'ASN':9,
         'GLN':10,
         'HIS':11,
         'HSD':11,
         'HSE':11,
         'SER':12,
         'THR':13,
         'VAL':14,
         'MET':15,
         'CYS':16,
         'NLE':17,
         'ARG':18,
         'LYS':19,
         'LEU':20,
         'ILE':21
         }

emb = np.array([AA2INT[x] for x in mol.resname], dtype='<U3')
np.save('data/chignolin_ca_embeddings.npy', emb)

print(
"""
2. Training

In the next step, we use coordinates and delta-forces to train the network.

SchNet architecture, applied here, learns the features using continuous filter convolutions on a graph neural network and predicts the forces 
and energy of the system.

A set of parameters in the configuration file train.yaml is listed here:

```
coords: "data/chignolin_ca_coords.npy"
forces: "data/chignolin_ca_deltaforces.npy"
embeddings: "data/chignolin_ca_embeddings.npy"
log_dir: train_light
num_epochs: 200
batch_size: 1024
lr: 0.0001
lr_patience: 4
lr_min: 1.e-06
lr_factor: 0.8
distributed_backend: dp
gpus: 4
num_nodes: 1
num_workers: 8
seed: 1
eval_interval: 1
save_interval: 1
progress: True
val_ratio: 0.05
test_ratio: 0.1
num_filters: 128
num_gaussians: 150
num_interactions: 3
max_z: 100
cutoff: 9
```

Now we will go through options in a configuration file:

    training input files locations are defined in parameters: coords, forces and embeddings
    log_dir - output folder
    lr - initial value of learning rate
    num_epochs - number of epochs run during the training
    batch_size - batch size
    lr - initial value of learning rate. The learning rate is optimized with torch.optim.lr_scheduler.ReduceLROnPlateau scheduler with parameters: lr_patience, lr_min and lr_factor
    distributed_backend - specifies distributed_backend pytorch-ligtning. Here dp (Data Parallel) is adjusted for training on multiple-gpus (gpus) and 1 machine (num_nodes). Other options include:
        Data Parallel (distributed_backend='dp')(multiple-gpus, 1 machine)
        DistributedDataParallel (distributed_backend=’ddp’) (multiple-gpus across many machines (python script based)).
        DistributedDataParallel (distributed_backend=’ddp_spawn’) (multiple-gpus across many machines (spawn based)).
        DistributedDataParallel 2 (distributed_backend=’ddp2’) (DP in a machine, DDP across machines).
        Horovod (distributed_backend=’horovod’) (multi-machine, multi-gpu, configured at runtime)
    gpus - number of GPUs used in training. Specified as a number of required units (eg. 4) or a list of cuda devices (eg. `[0, 2, 3]')
    num_nodes - number of machines used
    num_workers - number of workers in data loader
    seed for the calculation
    eval_interval - evaluation interval
    save_interval - saving interval
    progress - Progress bar during batching
    val_ratio - Percentual of validation set
    test_ratio - Percentual of test set
    Finally schnet-specific parameters:
        num_filters
        num_gaussians
        num_interactions
        max_z
        cutoff

Training is done using python script and can be run by a simple command:

```
mkdir train_light
python $PATH/torchmd-net/scripts/light_train.py -c train.yaml
```

where $PATH is the path to your torchmd-net repo.

The training saves 8 best epochs. The progress of the training is saved in TensorBoard session. 
The training and validation curves for the training of full Chignolin dataset are presented here:
"""
)



df = pd.read_csv('data/train_out_data.csv')

fig, ax = plt.subplots(nrows = 2, figsize = [8,8])
df.plot(x = 'epoch', y = ['train_loss', 'val_loss'], ax = ax[0])
ax[0].scatter([80], [df['val_loss'].loc[80]], s=200, c='r', marker=(5, 1), label='selected epoch')
ax[0].set_ylabel('Loss')
df.plot(x = 'epoch', y = 'lr', ax = ax[1])
ax[1].set_ylabel('learning rate')


