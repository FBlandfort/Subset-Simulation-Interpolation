

from scipy.stats import norm
from scipy.linalg import cholesky
from .. import main 
import numpy as np


#####
'''
The correlation coefficient for the Nataf transformation is evaluated, following:

Xiao, Qing. "Evaluating correlation coefficient for Nataf transformation." Probabilistic Engineering Mechanics 37 (2014): 1-6.


'''



def get_matched_corrX(random_attrs,corrX):
    '''
    match the correlation array to the corresponding random variables by its name, not necessary if typing in the correlation matrix in order
    '''
    if type(corrX)==int:
        return(np.array(np.eye(len(random_attrs.all))))
    matched_corrX=np.array(np.eye(len(random_attrs.all)))
    for match_1 in range(len(random_attrs.all)):
        for match_2 in range(len(random_attrs.all)):
            for corr_match in range(len(corrX)):
                if random_attrs.all[match_1].name==corrX[corr_match][0] and random_attrs.all[match_2].name==corrX[corr_match][1]:
                    matched_corrX[match_1,match_2]=corrX[corr_match][2]
                    matched_corrX[match_2,match_1]=corrX[corr_match][2]
    return(np.array(matched_corrX))





def g_cx(n,xi,xj,cz):
    '''
    cx(n,xi,xj,cz)
    ---------------------------------------------------
    n: number of points for hermite gaussian quadrature
    xi: rv object
    xj: rv object
    cz: correlation of rvs in Zroom
    --------------------------------------------------
    returns
    correlation of rvs in original room
    '''
    sum_ = [g_w(k,n)*g_w(l,n)*xi.ppf(p=norm.cdf(sqrt(2.0)*g_tk(k,n)))*xj.ppf(p=norm.cdf((cz*sqrt(2.0)*g_tk(k,n))+(sqrt(1.0-(cz**2.0))*sqrt(2.0)*g_tk(l,n)))) for k in range(n) for l in range(n)]
    sum_ = np.sum(sum_)
    cx=-(xi.mean*xj.mean)/(xi.sd*xj.sd) + (1.0/(pi*xi.sd*xj.sd))*sum_
    return(cx)




def g_tk(k,n):
    '''
    g_tk(k,n)
    kth root of nth hermite polynomial
    '''
    return(np.polynomial.hermite.hermroots(np.eye(n+1)[n])[k])


def g_w(k,n):
    '''
    g_w(k,n)
    kth hermite weight of nth hermite polynomial
    '''
    t_k=g_tk(k,n)
    k_w=((2.0**(n-1.0))*factorial(n)*sqrt(pi))/((n**2.0)*(np.polynomial.hermite.hermval(t_k,np.eye(n+1)[n-1])**2.0)) 
    return(k_w)



def get_cz(xi,xj,cx,nhermite=7,npol=9):
    '''
    use interpolation on g_cx func to get cz in standardard normal room for given cx and rvs xi xj
    '''
    #borders by lemma 1,2,3 in evaluating corr coeff for nataf transform, qing xiao
    if cx==0:
        return(0)
    border_l=-1.0
    border_u=1.0
    points=np.linspace(border_l,border_u,num=npol)
    cz_points=[]
    for pz in points:
        cz_points.append(float(g_cx(n=nhermite,xi=xi,xj=xj,cz=pz)))
    polynpol=interp1d(np.array(cz_points),points,fill_value="extrapolate")
    cz=polynpol(cx)
    return(cz)


def Z_by_U(corr_z,U):
    '''
    transforms U into Z by given correlation matrix in Z
    '''
    L=np.transpose(cholesky(corr_z)) 
    Z=np.dot(L,U)
    return(Z)
    

def U_sample(n):
    return(np.transpose(np.array(np.random.normal(loc=0.0,scale=1.0,size=n))))



def nataf(problem,U,corr_z=0): 
    '''
    Nataf transformation:
    returns a result object which has the corresponding U_samples, Z_samples, X_samples
    and its correlation matrix in Z
    if U_start not given, samples U by n_samples given
    '''
    if type(corr_z)==int: #have to compute the global variable, otherwise it is given already
        corr_z=problem.rvs.get_cz_col()    
        if np.all(np.linalg.eigvals(corr_z) > 0)==False: #cz not positive semidefinite, knowing it is symmetric so checking for eigenvalues suffices
            corr_z=np.array(nearPD(corr_z,nit=20))   #use Higham algorithm to get a close positive semidefinite matrix!
    Z_sample=Z_by_U(corr_z,U)
    X_sample=np.array([float(problem.rvs.all[i].ppf(p=norm.cdf(Z_sample[i])))
        for i in range(len(Z_sample))])
    result_nataf=main.result.result_obj()
    result_nataf.U=U
    result_nataf.Z=Z_sample
    result_nataf.X=X_sample
    result_nataf.corr_z=corr_z
    return(result_nataf)











############### Higham algorithm
'''
Code is a modified version of 
https://stackoverflow.com/questions/10939213/how-can-i-calculate-the-nearest-positive-semi-definite-matrix

and originates from:

Higham, Nicholas J. "Computing the nearest correlation matrix—a problem from finance." IMA journal of Numerical Analysis 22.3 (2002): 329-343.


'''

def getAplus(A):
    eigval, eigvec = np.linalg.eig(A)
    Q = np.array(eigvec)
    xdiag = np.array(np.diag(np.maximum(eigval, 0)))
    return(Q.dot(xdiag).dot(Q.T))

def getPs(A, W=None):
    W05 = np.array(W**.5)
    return(np.dot(np.linalg.inv(W05), getAplus(np.dot(W05,A,W05)), np.linalg.inv(W05)))

def getPu(A, W=None):
    Aret = np.array(A.copy())
    Aret[W > 0] = np.array(W)[W > 0]
    return(np.array(Aret))

def nearPD(A, nit=10):
    n = A.shape[0]
    W = np.identity(n) 
    deltaS = 0
    Yk = A.copy()
    for k in range(nit):
        Rk = Yk - deltaS
        Xk = getPs(Rk, W=W)
        deltaS = Xk - Rk
        Yk = getPu(Xk, W=W)
    return(np.asarray(Yk))












