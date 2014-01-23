# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 11:10:37 2012

@author: greg
"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import chain
import orthopoly as op
from colloc_rk import crk
from scipy.linalg import solve
import sys

if __name__ == '__main__':
    
    np.set_printoptions(precision=4,linewidth=999)
    
    # Array of layer thicknesses
    
    L = 100
    
    dl = np.array((L,L,L))
    
    # number of layers 
    ndl = len(dl)
    
    # Array of layer end points
    xl = np.hstack((0,np.cumsum(dl)))
    
    # requested element size
    dxlreq = 2.5
    
    # number of requested elements per layer
    nlreq = dl/float(dxlreq)
    
    # actual number of elements per layer
    nl = np.ceil(nlreq)
    
    ngpt = 1+int(sum(nl)) 
    
    # actual element thickness per layer
    dxl = dl/nl
    
    def elemrep(foo):
        return np.array(list(chain.from_iterable(map(lambda q: \
               np.tile(foo[q],nl[q]),xrange(ndl)))))
    
    
    # thickesses of every layer
    dx = elemrep(dxl)
    
    # elemental end points              
    xe = np.hstack((0,np.cumsum(dx)))             
    
    # Number of GRK stages per element
    nrk = 4
    
    # Legendre recursion coefficients 
    alpha,beta = op.rec_jacobi(nrk,0,0)
    
    # Gauss points on [-1,1]
    c,b = op.gauss(alpha,beta)
    
    # map to [-1,1]
    c = 0.5*(1+c)
    b = 0.5*b
    
    # Gauss Runge-Kutta coefficients
    A,b = crk(c)
    
    # compute the Gauss nodes on every element
    Xg = np.tile(xe[:-1],(nrk,1)).T + np.outer(dx,c)
    
    
    # Conduction band energy in every layer
#    Ecl = np.array((0.292,0))
    Ecl = np.array((0.292,0,0.292))
#    Ecl = np.array((0.292,0.292))    
    mel = np.array((0.096,0.067,0.096))
#    mel = np.array((0.096,0.067))
#    mel = np.array((0.096,0.096))
    
    Ec = elemrep(Ecl)
    me = elemrep(mel)
    
    E = float(sys.argv[1])
    
    Pg = np.tile(3.81/me,(nrk,1)).T
    
    Qg = np.tile(Ec-E,(nrk,1)).T
    
    
    yz = np.zeros((ngpt,2),dtype=complex)
    
    



    # compute the exact solution
    kL = np.sqrt(me[0]*(E-Ec[0]+0j)/3.81)    
    kR = np.sqrt(me[-1]*(E-Ec[-1]+0j)/3.81)    

    # Solve left to right

#    f1 = np.exp(-1j*kL*L)
#    f2 = -1j*f1*kL/me[0]

#    s11 = np.exp(1j*kR*L)    
#    s12 = np.exp(-1j*kR*L)
#    s21 = 1j*kR*s11/me[-1]
#    s22 = -1j*kR*s12/me[-1]
    
#    S = np.array(((s11,s12),(s21,s22)))    
    
#    f = np.array((f1,f2)) 
    
#    AB = solve(S,f)

#    exactL = np.exp(-1j*kL*xe)    
#    exactR = AB[0]*np.exp(1j*kR*xe)+AB[1]*np.exp(-1j*kR*xe)


    yz[0,:] = np.array((1,-3.81*1j*kL/me[0]))
   
    # Identity matrix
    I = np.identity(nrk)
    e = np.ones(nrk)
    O = np.zeros(nrk)
    
    
    for kk in xrange(ngpt-1):
        dxA = dx[kk]*A
        dxB = dx[kk]*b
    
        PA = dxA/np.tile(Pg[kk,:],(nrk,1))
        QA = dxA*np.tile(Qg[kk,:],(nrk,1))
        PB = dxB/Pg[kk,:]
        QB = dxB*Qg[kk,:]
    
        B = np.vstack((np.hstack((O,PB)),np.hstack((QB,O)))) 
        
        S = np.vstack((np.hstack((I,-PA)),np.hstack((-QA,I))))
        rhs = np.kron(yz[kk,:],e)    
        YZ = solve(S,rhs)
        yz[kk+1,:] = yz[kk,:] + np.dot(B,YZ)
    

    plt.subplot(211)    
    plt.plot(xe,np.abs(yz[:,0])**2,'-',lw=2)
#    plt.plot(xe,np.real(exactL)) 
#    plt.plot(xe,np.real(exactR)) 







    # Solve right to left

#    f1 = np.exp(1j*kR*L)
#    f2 = 1j*kR*f1/me[-1]
    
#    s11 = np.exp(1j*kL*L)
#    s12 = np.exp(-1j*kL*L)
#    s21 = 1j*kL*s11/me[0]
#    s22 = -1j*s12/me[0]

#    S = np.array(((s11,s12),(s21,s22)))    
#    f = np.array((f1,f2)) 
#    AB = solve(S,f)

#    exactR = np.exp(1j*kR*xe)    
#    exactL = AB[0]*np.exp(1j*kL*xe)+AB[1]*np.exp(-1j*kL*xe)
    
    yz[-1,:] = np.array((1,1j*3.81*kR/me[-1]))

    # Integrate from right to left
    
    for kk in xrange(ngpt-1):
        
        jj = ngpt-kk-2
                
        dxA = -dx[jj]*A
        dxB = -dx[jj]*b
    
        PA = dxA/np.tile(Pg[jj,:],(nrk,1))
        QA = dxA*np.tile(Qg[jj,:],(nrk,1))
        PB = dxB/Pg[jj,:]
        QB = dxB*Qg[jj,:]
    
        B = np.vstack((np.hstack((O,PB)),np.hstack((QB,O)))) 
        
        S = np.vstack((np.hstack((I,-PA)),np.hstack((-QA,I))))
        rhs = np.kron(yz[jj+1,:],e)    
        YZ = solve(S,rhs)
        yz[jj,:] = yz[jj+1,:] + np.dot(B,YZ)


    plt.subplot(212)    
    plt.plot(xe,np.abs(yz[:,0])**2,'-',lw=2)
#    plt.plot(xe,np.real(exactL)) 
#    plt.plot(xe,np.real(exactR)) 

    
    plt.show()    
    
    
    
    
    
    
    
    
    
    
    
    
