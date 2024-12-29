TOC
- [Quick start on Twnia-2](#quick-start-on-taiwan-2)
- [Quick start on H100](#quick-start-on-h100-cluster)
- [Venv detail on H100](#env-on-h100-cluster)



## Quick start on Taiwan-2
We use the system bundled NVHPC 24.11 and self-built Python 3.12.8. To use them, first load my module path 
```
module use /opt/ohpc/pkg/kagra/modulefiles     # this line can be added in your ~/.bashrc
ml sys/nvhpc-hpcx
ml python/3.12.8
```

Based on that, a ready-to-use virtual env for running cuQuantum/24 can be loaded by
```
source /opt/ohpc/pkg/kagra/ENV/cuquantum24/bin/activate
```

Instead, if you want to build your venv, just follow this:
```
ml purge
ml nvhpc-24.11_hpcx-2.20_cuda-12.6
ml python/3.12.8

ENV=<Your venv path>
python3 -m venv ${ENV}

pip install mpi4py
pip install seaborn pandas scikit-learn
pip install cuquantum-python-cu12 qiskit_aer-gpu cupy-cuda12x qiskit qiskit_algorithms qiskit_machine_learning
```

## Quick start on H100 cluster
I built OpenMPI based on prebuilt packages in NVHPC (UCX, UCC, Hcoll, NCCL, CUDA SDK, ...) and Python 3.12 by gcc. To use them, first load the module path 
```
module use /work/p00lcy01/pubmodules           # this line can be added in your ~/.bashrc
ml openmpi/5.0.6
ml python/3.12.8
```

Based on that, a ready-to-use virtual env for running cuQuantum/24 can be loaded by
```
source /work/p00lcy01/venv/cuq24_py312/bin/activate
```

Instead, if you want to build your venv, just follow this:
```
ml purge
ml python/3.12.8
ml openmpi/5.0.6

ENV=<Your venv path>
python3 -m venv ${ENV}

pip install mpi4py
pip install seaborn pandas scikit-learn
pip install cuquantum-python-cu12 qiskit_aer-gpu cupy-cuda12x qiskit qiskit_algorithms qiskit_machine_learning
```

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
OpenMPI 5.0.5 build with NCHPC and HPC-X. Note that some mpi4py test need UCX-mt. The built script are
```
ml purge
ml pmix hwloc
ml nvhpc-hpcx-cuda12/24.7
ml hpcx-mt

export CUDA_PATH=${NVHPC_ROOT}/cuda
export CUDA_HOME=${NVHPC_ROOT}/cuda
export CUDA_ROOT=${NVHPC_ROOT}/cuda

PKG_ROOT=/work/p00lcy01/pkg

CC=nvc++ CXX=nvc++ FC=nvfortran \
CFLAGS=-fPIC CXXFLAGS=-fPIC FCFLAGS="-Mstandard -fPIC" \
../configure --prefix=${PKG_ROOT}/openmpi-5.0.6_mt  \
             --with-slurm=/usr --enable-mpi-fortran \
             --with-knem=/opt/knem-1.1.4.90mlnx3 \
             --with-pmix=${PMIX_DIR} --with-hwloc=${HWLOC_DIR} \
             --with-cuda=${CUDA_PATH} --with-cuda-libdir=/usr/lib64 \
             --with-ucx=${HPCX_UCX_DIR} \
             --with-ucc=${HPCX_UCC_DIR} \
             --with-hcoll=${HPCX_HCOLL_DIR} \
             --with-platform=../contrib/platform/mellanox/optimized
```

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

