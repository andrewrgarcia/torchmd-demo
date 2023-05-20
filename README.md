# torchmd-demo

Tests for torchmd https://github.com/torchmd/torchmd

## torchmd-cg

Tutorial from https://github.com/torchmd/torchmd-cg/blob/master/tutorial/Chignolin_Coarse-Grained_Tutorial.ipynb
transcribed to `main.py` file. 

### Installation

```bash
# install torchmd and dependencies
conda create -n torchmd
conda activate torchmd
conda install mamba python=3.10 -c conda-forge
mamba install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia -c conda-forge

conda install pyyaml ipython scikit-learn tqdm

mamba install moleculekit parmed jupyter -c acellera -c conda-forge # For running the examples

pip install pytorch-lightning
pip install torchmd
pip install moleculekit


conda install seaborn pandas jupyter

# clone torchmd_cg https://github.com/torchmd/torchmd-cg
cd ~/
git clone https://github.com/torchmd/torchmd-cg.git
cp -r torchmd-cg/torchmd_cg torchmd-tests
cd torchmd-tests

wget pub.htmd.org/torchMD_tutorial_data.tar.gz
tar -xvf torchMD_tutorial_data.tar.gz
```

Then run `python main.py` 
