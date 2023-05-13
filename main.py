import os
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm
from scipy.optimize import curve_fit
from moleculekit.molecule import Molecule
from moleculekit.projections.metricrmsd import MetricRmsd

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")



from torchmd_cg.utils.psfwriter import pdb2psf_CA

PDB_file = 'data/chignolin_cln025.pdb'
PSF_file = 'data/chignolin_ca_top.psf'

pdb2psf_CA(PDB_file, PSF_file, bonds = True, angles = False)

mol = Molecule('data/chignolin_ca_top.psf')
arr = np.load('data/chignolin_ca_coords.npy')


mol.coords = np.moveaxis(arr, 0, -1) 


priors = {}
priors['atomtypes'] = list(set(mol.atomtype))
priors['bonds'] = {}
priors['lj'] = {}
priors['electrostatics'] = {at: {'charge': 0.0} for at in priors['atomtypes']}
priors['masses'] = {at: 12.0 for at in priors['atomtypes']}




from torchmd_cg.utils.prior_fit import get_param_bonded

T = 350 # K
fit_range = [3.55,4.2]

bond_params = get_param_bonded(mol, fit_range, T)

priors['bonds'] = bond_params

