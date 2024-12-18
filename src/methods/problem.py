import numpy as np
import os
from helpers import random_matrix

A_PATH = "A_4x100.txt"
P_PATH = "P_4x100.txt"
N = 4
M = 100
X0 = np.zeros(N)
MAX_ITERATION = 100 # multiples of N
t_bar = 0.5

# SETS UP A, P, T
try: 
    A = np.loadtxt(A_PATH, dtype="float")
    P = np.loadtxt(P_PATH, dtype="float")
except: 
    print(f"WARNING: Given Path for A, P matrices for the task assignment problem do not exist. Random matrices generated({M}x{N}).")
    A = random_matrix(M, N)
    P = random_matrix(M, N)

# MAKES TIME VECTOR
T = []
for j in range(N):
    j_time = 0
    for i in range(M):
        j_time += P[i, j]
    T.append(t_bar*j_time/N)
T = np.array(T)


def f_i(x, i):
    min_obj = np.inf
    min_j = -1
    for j in range(N):
        obj_j = A[i, j]+x[j]*P[i, j]
        if min_obj > obj_j:
            min_obj = obj_j
            min_j = j
    return (min_obj - float(np.dot(T, x))/M, min_j)
    

def f(x):
    obj = 0
    for i in range(M):
        obj_i, _ = f_i(x, i)
        obj += obj_i
    return obj


def g_i(i, min_j):
    subgrad = []
    for j in range(N):
        if j == min_j:
            subgrad.append(P[i, j] - T[j]/M)
        else:
            subgrad.append(-T[j]/M)
    return np.array(subgrad)