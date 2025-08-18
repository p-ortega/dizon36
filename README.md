# DIZON 36  

**Dizon 36** is a sandbox reactive transport model designed to benchmark the [`mf6rtm`](https://github.com/p-ortega/mf6rtm) code and to explore decision-support workflows for reactive transport modeling using machine-learning surrogates.  

## Workflow
The main workflow is controlled through boolean flags. Toggle these to enable or skip steps:  

```python
prep_obs = True   # prepare observation CSVs  
run_base = True   # build and run a base model  
prep_pest = False # prepare PEST++ setup  
run_pest = False  # run PEST++ calibration/uncertainty analysis  
```

## Plots
A companion Jupyter Notebook is included for generating plots. Feel free to edit and adapt it for your own analyses.

## Dependencies  

This project relies on:  
- [pestpp](https://github.com/usgs/pestpp)  
- [pyemu](https://github.com/pypest/pyemu)  
- [mf6rtm](https://github.com/p-ortega/mf6rtm/tree/feature/externalio) — use the `externalio` branch  

To install the required environment, use the provided `environment.yml`:  

```bash
conda env create -f environment.yml
conda activate dizon36
```

