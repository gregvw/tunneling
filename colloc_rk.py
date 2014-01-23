import numpy as np
from scipy.linalg import solve

def crk(c):
    """ Compute the Collocation Runge-Kutta coefficients A and b
        given an array of collocation points c on [0,1] """

    n = c.size                      # Count nodes
    theta = np.arccos(2*c-1)        # Map to [0,pi]
    k = np.arange(n+1)              
    T = np.cos(np.outer(theta,k))   # Chebyshev Vandermonde on given nodes
    vals = 1/(2.0*(1+k))            # Integrated Chebyshev polynomial coeffs
    band = np.hstack((0, vals)) 
    Bt = np.diag(band[1:-1],1)-np.diag(band[:n],-1)
    Bt[0,1] = 1
    BTi = solve(T[:,:n].T,Bt[:n,:]).T
    e = np.ones(n+1)
    U = np.vstack((1-2*(k % 2),T,e))
    UBTi = np.dot(U,BTi)
    M = UBTi-np.outer(np.ones(n+2),UBTi[0,:])
    A = M[1:-1,:]/2
    b = M[-1,:]/2  
    return A, b


def rkstab(A,b,xmin,xmax,ymax,n):
    """ Evaluate the stability region of a Runge-Kutta method
        given its coefficients A and b
    """
    
    d = b.size
    m = int(0.5*n)
    x = np.linspace(xmin,xmax,n)
    y = np.linspace(0,ymax,m)
    X, Y = np.meshgrid(x, y)  
    z = X+1j*Y 
    I = np.identity(d)     
    e = np.ones(d)
    r = np.zeros((m,n),dtype=complex)

    for ix in xrange(n):
        for iy in xrange(m):
            s = solve(I-z[iy,ix]*A,e)
            r[iy,ix] = 1 + z[iy,ix]*np.dot(b,s)

    R = np.abs(r)
    X = np.vstack((np.flipud(X[1:,:]),X))    
    Y = np.vstack((-np.flipud(Y[1:,:]),Y))    
    R = np.vstack((np.flipud(R[1:,:]),R))
    return X,Y,R


if __name__ == '__main__':
    """ Compute stability region for 3-point Radau """
    import matplotlib.pyplot as plt
    import orthopoly as op

    alpha,beta = op.rec_jacobi(3,0,0)
    x,w = op.radau(alpha,beta,1)
    # x,w = op.gauss(alpha,beta)
    
    c = 0.5*(1+x)

    A,b = crk(c)
    x,y,r = rkstab(A,b,-15,15,15,100)

    plt.contourf(x,y,r,np.array((0,1,1e16)),colors = ('b','r'))
    plt.title('Stability of 3-pt Radau')
    plt.show()



