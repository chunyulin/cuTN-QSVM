## ENV on H100 cluster
### Job script
```
#SBATCH --nodes=2 --ntasks-per-node=8 --cpus-per-task=12 --gres=gpu:8
ml purge
ml python/3.12.8
ml openmpi/5.0.6
ENV=/work/p00lcy01/venv/cuq24_py312
source ${ENV}/bin/activate
export UCX_PROTO_ENABLE=n
srun --mpi-pmix python banchmark_qsvm_tnsm-mpi.py
```
### OpenMPI 5.0.5
- OpenMPI 5.0.5 build with NCHPC and HPC-X. Note that some mpi4py test need UCX-mt

### Python 3.12.8
- Python 3.12.8 built with default GCC. (Python 3.13 fail to install mpi4py?!)

### Create python venv
```
## on h100 cluster
ENV=/work/p00lcy01/venv/cuq24_py312
python -m venv ${ENV}
pip install --upgrade pip
pip install mpi4py seaborn pandas scikit-learn
pip install cuquantum-python-cu12 qiskit_aer-gpu cupy-cuda12x qiskit qiskit_algorithms qiskit_machine_learning
```

