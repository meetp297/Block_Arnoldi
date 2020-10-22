from scipy import linalg
import numpy as np
import random
from numpy import random

A=random.rand(50,50)
B=random.rand(50,10)
def arnoldi_extra_space(A,B,m_1):
'''
input:
      A:matrix(N*N) mentioned in krylov subspace
      B:matrix(N*O) mentioned in krylov subspace
      m_1:number of krylov iteration
output:
      Q:matrix(N*m_1p) orthogonal basis of krylov subspace
      H:matrix(m_1p*m_1p) coefficient matrix 
      where p=rank of q_0
'''

    q_0,r=linalg.qr(B,mode='economic')
    q=q_0
    p=np.linalg.matrix_rank(q)
    h=[[] for i in range(m_1)]
    for j in range(0,m_1):
        t=A@q[:,p*j:p*j+p]
        for i in range(j+1):
            h_new=(np.array(q[:,p*i:p*i+p])).T @ t
            t=t-np.array(q[:,p*i:p*i+p]) @ h_new
            if len(h[j]):
                h[j]=np.concatenate((h[j],h_new),axis=0)
            else:
                h[j]=h_new
        q_new,h_new=linalg.qr(t,mode='economic')
        q=np.concatenate((q,q_new),axis=1)
        h[j]=np.concatenate((h[j],h_new),axis=0)
    
    #here we are using extra space to convert h[j]'s into matrix form    
    def convert_H(H):
        l,b=H[-1].shape
        h=np.concatenate((H[0],np.zeros((l-H[0].shape[0],b),dtype='float')),axis=0)
        for i in range(1,len(H)):
            h_temp=np.concatenate((H[i],np.zeros((l-H[i].shape[0],b),dtype='float')),axis=0)
            h=np.concatenate((h,h_temp),axis=1)
        return h
    H=convert_H(h)
    return q[:,:-p],H[:-p,:]
Q,H=arnoldi(A,B,30)
np.allclose(A@Q,Q@H)

def arnoldi_no_extra_space(A,B,m_1):
'''
input:
      A:matrix(N*N) mentioned in krylov subspace
      B:matrix(N*O) mentioned in krylov subspace
      m_1:number of krylov iteration
output:
      Q:matrix(N*m_1p) orthogonal basis of krylov subspace
      H:matrix(m_1p*m_1p) coefficient matrix 
      where p=rank of q_0
'''
      t=A@q[:,p*0:p*0+p]
      h_new=(np.array(q[:,p*0:p*0+p])).T @ t
      t=t-np.array(q[:,p*0:p*0+p]) @ h_new
      h=h_new
      q_new,h_new=linalg.qr(t,mode='economic')
      q=np.concatenate((q,q_new),axis=1)
      h=np.concatenate((h,h_new),axis=0)
      for j in range(k+1,k+m_1):
            t=A@q[:,p*j:p*j+p]
            h1=[]
            for i in range(j+1):
                  h_new=(np.array(q[:,p*i:p*i+p])).T @ t
                  t=t-np.array(q[:,p*i:p*i+p]) @ h_new
                  if len(h1):
                        h1=np.concatenate((h1,h_new),axis=0)
                  else:
                        h1=h_new
            q_new,h_new=linalg.qr(t,mode='economic')
            q=np.concatenate((q,q_new),axis=1)
            h1=np.concatenate((h1,h_new),axis=0)
            l,b=h1.shape
            h=np.concatenate((h,np.zeros((l-h.shape[0],h.shape[1]),dtype='float')),axis=0)
            h=np.concatenate((h,h1),axis=1)
      return q[:,:-p],H[:-p,:]
Q,H=arnoldi(A,B,30)
np.allclose(A@Q,Q@H)

      
      
      
      
      

