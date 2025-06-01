### Initial setup
1. Clone the repository:
```bash
git clone https://github.com/savelovme/adaptive-mcmc.git
cd adaptive-mcmc
```

2. Get the `PosteriorDB`:
```bash
git clone https://github.com/stan-dev/posteriordb.git
```

3. Create conda environment:
```bash
conda env create -f python/environment.yml
conda activate mcmc
```

### Reproducing the results of the numerical experiments
1. Set environment variables:
```bash
export MCMC_WORKDIR=$(pwd)
export PYTHONPATH=$MCMC_WORKDIR/python:$PYTHONPATH
```

2. Run scripts from `python/scripts` (some may take multiple hours):
```bash
cd $MCMC_WORKDIR/python/scripts
python <script_name>.py
```
3. Run jupyter notebooks to explore the runs and produce images:
```bash
python -m notebook --notebook-dir $MCMC_WORKDIR/python/jupyter
```