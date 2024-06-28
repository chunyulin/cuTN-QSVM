import cuquantum, sklearn, qiskit, qiskit_algorithms
print(f"Qiskit: {qiskit.__version__}")
print(f"Qiskit ML: {qiskit_algorithms.__version__}")
print(f"cuQuantum: {cuquantum.__version__}")
print(f"skLearn: {sklearn.__version__}")

import numpy as np
import cupy as cp
import pandas as pd
import time
import matplotlib.pyplot as plt
from itertools import combinations,product
from multiprocessing import Pool
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.datasets import load_digits, fetch_openml
from sklearn.model_selection import GridSearchCV
from qiskit import QuantumCircuit
from qiskit_aer import Aer
from qiskit.circuit import ParameterVector
from cuquantum import *
from cupy.cuda.runtime import getDeviceCount
from mpi4py import MPI

root = 0
comm = MPI.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()
device_id = 0
cp.cuda.Device(device_id).use()
print(f"Using GPU:{device_id}")

mnist = fetch_openml('mnist_784')
X = mnist.data.to_numpy()
Y = mnist.target.to_numpy().astype(int)
class_list = [7,9]
c01 = np.where((Y == class_list[0])|(Y == class_list[1]))
X,Y = X[c01],Y[c01]
MAX=1000
data_train, label_train = X[:MAX],Y[:MAX]
X_train, X_val, Y_train, Y_val = train_test_split(data_train, label_train, test_size = 0.2, random_state=255)
print("Train-set shape:", np.shape(X_train), np.shape(Y_train))   ## 800 x 28x 28
print("Test -set shape:", np.shape(X_val), np.shape(Y_val))
print("Train:", Y_train)
print("Val:", Y_val)

def data_prepare(n_dim, sample_train, sample_test, nb1, nb2):
    std_scale = StandardScaler().fit(sample_train)
    data = std_scale.transform(sample_train)
    sample_train = std_scale.transform(sample_train)
    sample_test  = std_scale.transform(sample_test)
    #pca = PCA(n_components=n_dim, svd_solver="full").fit(data)
    pca = PCA(n_components=n_dim, svd_solver="auto").fit(data)    ## use auto for faster checking
    sample_train = pca.transform(sample_train)
    sample_test = pca.transform(sample_test)
    samples = np.append(sample_train, sample_test, axis=0)
    minmax_scale = MinMaxScaler((-1, 1)).fit(samples)
    sample_train = minmax_scale.transform(sample_train)[:nb1]
    sample_test = minmax_scale.transform(sample_test)[:nb2]
    return sample_train, sample_test

def make_bsp(n_dim):
    param = ParameterVector("p",n_dim)
    bsp_qc = QuantumCircuit(n_dim)
    bsp_qc.h(list(range(n_dim)))
    i = 0
    for q in range(n_dim):
        bsp_qc.rz(param.params[q],[q])
        bsp_qc.ry(param.params[q],[q])
    for q in range(n_dim-1):
        bsp_qc.cx(0+i, 1+i)
        i+=1
    for q in range(n_dim):
        bsp_qc.rz(param.params[q],[q])
    return bsp_qc

def kernel_matrix_tnsm(y_t, x_t, opers, indices_list, network, mode=None):
    kernel_matrix = np.zeros((len(y_t),len(x_t)))
    i = -1
    with network as tn:
        for i1, i2 in indices_list:
            i += 1
            tn.reset_operands(*opers[i])
            amp_tn = abs(tn.contract()) ** 2
            kernel_matrix[i1-1][i2-1] = np.round(amp_tn,8)
    if mode == 'train':
        kernel_matrix = kernel_matrix + kernel_matrix.T+np.diag(np.ones((len(x_t))))
    return kernel_matrix

def run_tnsm(n_dim, nb1, nb2):
    a = str(0).zfill(n_dim)
    data_train, data_val  = data_prepare(n_dim, X_train, X_val, nb1, nb2)
    indices_list_t = list(combinations(range(1, len(data_train) + 1), 2))

    ### TN expression from the first element
    psix=make_bsp(n_dim)
    psiy=psix.inverse()
    exp_t = -1
    
    ### Buing operators for each kernel element
    t0 = time.time()
    oper_train = []
    for i1, i2 in indices_list_t:
        cir = psix.assign_parameters(data_train[i1-1]).compose(psiy.assign_parameters(data_train[i2-1]))
        con = CircuitToEinsum(cir, dtype='complex128', backend='cupy')
        exp, oper = con.amplitude( a )
        oper_train.append(oper)
    oper_t = round((time.time()-t0),3)

    ### Find optimal contraction path
    t0 = time.time()
    options = NetworkOptions(blocking="auto",device_id=device_id)
    oper = oper_train[0]
    network = Network(exp, *oper,options=options)
    path, info = network.contract_path()
    network.autotune(iterations=20)
    path_t = round((time.time()-t0),3)

    ### Calculte kernels
    t0 = time.time()
    tnsm_kernel_matrix_train = kernel_matrix_tnsm(data_train, data_train, oper_train, indices_list_t, network, mode='train')
    tnsm_kernel_t = round((time.time()-t0),3)
    print(n_dim, exp_t, oper_t, path_t, tnsm_kernel_t, len(oper_train))
    print(tnsm_kernel_matrix_train)

#run_tnsm(3,3,2)
#run_tnsm(4,4,2)
#run_tnsm(10,2,1)
for d in [2,5,10,50,100,500,1000]:
    run_tnsm(128,d,1)

# for q in range(2,34):
#     run_tnsm(q,2,1)
# for q in range(42,200,10):
#     run_tnsm(q,2,1)
# for q in [200,300,400,500,600,784]:
#     run_tnsm(q,2,1)
