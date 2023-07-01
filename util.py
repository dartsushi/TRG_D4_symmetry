import scipy
from opt_3leg import S_to_M, S_to_N
import numpy as np
from ncon import ncon
from tqdm import tqdm

def find_projector(A):
    mat = transfer_matrix(A, A.transpose(1,2,3,0))
    shape = mat.shape
    mat = mat.reshape(shape[0]*shape[1],-1)
    # val, vec = np.linalg.eigh(mat)
    val, vec = scipy.sparse.linalg.eigsh(mat,which="LM",k=3)
    vec = vec[:,np.argsort(val)[::-1]]
    val = np.sort(val)[::-1]
    #print(np.log(val[0]/val[1])/2/np.pi)
    vec = vec.reshape(shape[2],shape[3],-1)
    return vec

def gauge_fix(S):
    T = np.zeros((3,3,3))
    M = S_to_M(S)
    A = np.einsum("ijkl,nkjm->mnli",M,M)
    A_before = S_to_N(S)
    # print("Finding the projectors...")
    P1 = find_projector(A_before)
    P2 = find_projector(A)
    M_new = ncon([M,P1],([-1,1,2,-3],[1,2,-2]))
    # print("Computing the matrix elements...")
    # Memory cost is lower
    for i in range(3):
        for j in range(3):
            for k in range(3):          
                T[i,j,k] = np.einsum("ii",M_new[:,i]@M_new[:,j]@P2[:,:,k].T)
    return T/T[0,0,0]

def combine_M(Mp,i,j,k,l):
    return np.einsum("ii",Mp[:,i]@Mp[:,j]@Mp[:,k]@Mp[:,l])


def fp_tensor(S):
    M = S_to_M(S)
    A_before = S_to_N(S)
    P1 = find_projector(A_before)
    if P1.shape[2] > 3:
        P1 = P1[:,:,:3]
    M_new = ncon([M,P1],([-1,1,2,-3],[1,2,-2]))
    T = np.zeros((3,3,3,3))
    for i in tqdm(range(3)):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    T[i,j,k,l] = combine_M(M_new,i,j,k,l)
    T/=T[0,0,0,0]
    return T

#Create a 2x2 transfer matrix in y-direction. 
def transfer_matrix(tA, tB, x_direction = False):
    if x_direction:
        row = np.einsum("ijkl,nlmj->mkni",tA,tB)
    else:
        row=np.einsum("ijkl,kmin->jmln",tA,tB)
    row=np.einsum("ijkl,lknm->ijmn",row,row)
    return row