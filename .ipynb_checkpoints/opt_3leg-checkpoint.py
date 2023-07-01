import numpy as np
from ncon import ncon
from copy import copy
from scipy.optimize import minimize
import entanglement_filtering as ef


#Create a 2x2 transfer matrix in y-direction. 
def transfer_matrix(tA, tB):
    row=np.einsum("ijkl,kmin->jmln",tA,tB)
    row=np.einsum("ijkl,lknm->ijmn",row,row)
    return row

# Normalization of the tensors and their norm 
def normalize_tensor(tA, tB):
    norm = transfer_matrix(tA, tB)
    norm = np.trace(norm, axis1=0, axis2=2)
    norm = np.trace(norm, axis1=0, axis2=1)
    sitenorm = norm**(1/4)
    return tA / sitenorm, tB / sitenorm, norm

def one_loop_projector(phi, pos, d_cut):
    # Initializaiton with one-loop 
	# L. Wang and F. Verstraete arXiv:1110.4362
    L = np.identity(np.shape(phi[0])[0])
    for n in range(pos + 1):
        L = ef.QR_L(L, phi[n])
    R = np.identity(np.shape(phi[-1])[-1])
    for n in range(len(phi), pos + 1, -1):
        R = ef.QR_R(R, phi[n - 1])
    args = {'bondm': d_cut}
    return ef.P_decomposition(R,L,args,mode='bondm')

# def SVD_filter(tA, d_cut):
#     psiA = ef.make_psi(tA, tA.transpose(1,2,3,0), reshape=False)
#     psi = ef.make_psi(tA, tA.transpose(1,2,3,0))
#     psiB = []
#     # filter phi for one-loop and insert the projector
#     for i in range(1):
#         s1, s2 = SVD12(psiA[i], 2*d_cut)
#         phi = copy(psi)
#         del phi[i]
#         phi.insert(i, s1)
#         phi.insert(i + 1, s2)
#         PR, PL = one_loop_projector(phi, i, d_cut)
#     return np.tensordot(s1, PR, axes=1),np.tensordot(PL, s2, axes=1)

def SVD12(A, d_cut,cutoff = 1e-14):
    shape = A.shape
    mat = A.reshape(shape[0]*shape[1],-1)
    u,s,vh = np.linalg.svd(mat,full_matrices=True)
    s = s[s>cutoff]
    
    chi = min(d_cut, len(s))
    sq_s = np.diag(np.sqrt(s[:chi]))
    s1 = np.dot(u[:,:chi],sq_s).reshape(shape[0],shape[1],-1)
    s2 = np.dot(sq_s,vh[:chi,:]).reshape(-1,shape[2],shape[3])
    return s1, s2

def eigh12(A, d_cut,cutoff = 1e-14):
    shape = A.shape
    mat = A.transpose(0,1,3,2).reshape(shape[0]*shape[1],-1)
    val,vec = np.linalg.eigh(mat)
    ind = np.argsort(val)[::-1]
    u = vec[:,ind]
    s = val[ind]
    s = s[s>cutoff]
    
    chi = min(d_cut, len(s))
    sq_s = np.diag(np.sqrt(s[:chi]))
    s1 = np.dot(u[:,:chi],sq_s).reshape(shape[0],shape[1],-1)
    s2 = np.dot(sq_s,u[:,:chi].T).reshape(-1,shape[3],shape[2])
    return s1, s2.transpose(0,2,1)

def S_to_N(S):
    return np.einsum("ijs,lks->ijkl",S,S)

def S_to_M(S):
    return np.einsum("ijk,iml->lmjk",S,S)

def innerMs(M1,M2):
    T= np.einsum("ijkl,mjkn->imln",M1,M2)
    T = T.reshape(T.shape[0]*T.shape[1], -1)
    for i in range(2):
        T = T @ T
    return np.einsum("ii",T)

# args = [tA, innerMs(tA,tA)]
def cost_C4(S,args):
    N = S_to_N(S)
    A = args[0]
    cost_AA = args[1]
    cost_AN = innerMs(A, N)
    cost_NN = innerMs(N,N)
    return cost_AA+cost_NN-2*cost_AN

# δ<M1|M2>/δS2
def grad_innerproduct(M1,S2):
    N = S_to_N(S2)
    T= np.einsum("ijkl,mjkn->imln",M1,N)
    shape = T.shape
    T = T.reshape(shape[0]*shape[1], -1)
    mat = T @ T @ T
    mat = mat.reshape(shape[0],shape[1],shape[2],shape[3])
    return np.einsum("ijkl,kmni,lmo->jno",mat, M1,S2)

"""
i --mat-- k ---- M1 ---i
      |                |     |
      |               m   n
j -------- l ---S2 --o    
"""
def grad_cost(S,args):
    A = args[0]
    N = S_to_N(S)
    return 8*(grad_innerproduct(N,S)-grad_innerproduct(A,S))

def optimize_S(S, args, tol = 1e-8):
    cost = np.inf
    loop_counter = 0
    S_new = S
    cost = cost_C4(S_new,args=args)
    cost_min = cost
    S_return = S_new
    epsilon = 1
    while loop_counter < 1000 and cost > 1e-8:
        grad = grad_cost(S_new,args=args)
        grad_norm = np.linalg.norm(grad)
        delta = cost/(grad_norm**2)
        S_new -=  delta*grad*epsilon
        loop_counter += 1
        epsilon*=0.98
        cost = cost_C4(S_new,args=args)
        if cost < cost_min:
            S_return = S_new
    return S_return

# args = [tA, innerMs(tA,tA)]
def cost_C4_flat(S_flat,args):
    A = args[0]
    cost_AA = args[1]
    shape = A.shape
    N = S_to_N(S_flat.reshape(shape[0],shape[1],-1))
    cost_AN = innerMs(A, N)
    cost_NN = innerMs(N,N)
    return cost_AA+cost_NN-2*cost_AN

def grad_cost_flat(S_flat,args):
    A = args[0]
    shape = A.shape
    S = S_flat.reshape(shape[0],shape[1],-1)
    N = S_to_N(S)
    return 16*(grad_innerproduct(N,S)-grad_innerproduct(A,S)).flatten()

# Conjugate Gradient
def optimize_CG(S, args, maxiter = 400):
    S_flat = S.flatten()
    result = minimize(fun = cost_C4_flat,
                  x0=S_flat, 
                  args = (args,),
                  method="CG",
                  jac = grad_cost_flat,
                  options={'disp': True,'maxiter':maxiter}
                 )
    return (result.x).reshape(S.shape)

