# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 18:25:32 2017

@author: mahbo

"""

import warnings, numpy as np, scipy.linalg as la
from gurMod import *
from mosSVM import *
from mosPCA import *
from mosPCAMult import *
from sklearn.covariance import MinCovDet
warnings.filterwarnings('ignore')

def maxROC(m,dat,rsp,pred=None,perc=1):
    if pred is None: pred = m.pred(dat)
    predSort = np.argsort(pred.flatten()).astype(int)
    rsp = rsp[predSort]
    sum1 = max(sum(rsp),1)
    sum0 = max(sum(1-rsp),1)
    rsp1 = np.flip(np.cumsum(rsp),axis=0)
    rsp0 = np.flip(np.cumsum(1-rsp),axis=0)
    brange = np.sort(np.random.choice(range(len(pred)),int(perc*len(pred)),replace=False))
    roc = np.abs(rsp1[brange]/sum1-rsp0[brange]/sum0)
    return max(roc), pred[predSort[brange[np.argmax(roc)]]]

class model():
    # A model object is the conduit through which all optimization is done. It stand to
    # handle the different types of optimization objects from a unified framework from
    # the perspective of the problem object. It has the added benefit of being a self
    # contained optimization, allowing for cross-validation or comparisons at the level
    # of the problem object without having to define all of the extra variables and
    # constraint coefficients associated with each individual optimization
    def __init__(self, dat, rsp=None, lam=1, conic=True, dual=False, kernel=None, isPCA=False, dimPCA=None, outputFlag=False):
        self.isGur = False
        self.numPoints = dat.shape[0]
        self.numFields = dat.shape[1]
        self.isPCA = isPCA
        if dual and self.isGur: print('Dual problem not available when using Gurobi')
        if kernel is None: kernel = lambda x,y: x.T.dot(y)
        self.kernel = kernel
        self.Status = "unsolved"
        self.RunTime = 0
        self.rsp = rsp
        self.conic = conic
        self.dual = dual
        self.B = np.zeros((self.numFields,1))
        self.b = 0
        self.ObjVal = 0
        self.numProjCons = 0
        self.K = None
        self.Y = None
        self.dimPCA = dimPCA
        if self.dual:
            self.K = np.empty((self.numPoints,self.numPoints))
            self.K[np.tril_indices(self.numPoints)] = [kernel(dat[i],dat[j]) for i in range(self.numPoints) for j in range(i+1)]
            self.K += np.tril(self.K,-1).T
            self.Y = np.array([(2*y1-1)*(2*y2-1) for y1 in rsp for y2 in rsp]).reshape((len(rsp),len(rsp)))
            if np.sum(np.isnan(self.K))>0:
                self.K[np.where(np.isnan(self.K))] = 0.9995
            if np.sum(np.isinf(self.K))>0:
                print(np.where(np.isinf(self.K)))
        if isPCA:
            if dimPCA is None or dimPCA==1: self.m = mosPCA(dat,outputFlag=outputFlag)
            else: self.m = mosPCAMult(dat, dimPCA, outputFlag=outputFlag)
        else:
            self.m = mosSVM(rsp,dat,lam,conic,dual,self.K,self.Y,outputFlag=outputFlag)
        
    def kFold(self, k=5):
        # Splits data into k folds
        idx = np.arange(self.numPoints)
        np.random.shuffle(idx)
        folds = [idx[int(i*self.numPoints/k):int((i+1)*self.numPoints/k)] for i in range(k)]
        return folds
    
    def optimize(self, outputFlag=False) -> None:
        # Runs the optimization procedure
        self.m.optimize()
        self.RunTime = self.m.m.RunTime if self.isGur else self.m.RunTime
        self.B = np.array(self.m.getB())
        if len(self.B.shape)==1: self.B = self.B.reshape((len(self.m.getB()),1))
        if not self.isPCA:    
            self.alpha = np.array(self.m.getAlpha()).reshape((self.numPoints,1))
            self.b = self.m.getb()
        self.ObjVal = self.m.getObj()
        if outputFlag: print("Optimization time: %s" % (round(self.RunTime,2)))
    
    def getStatus(self):
        # Returns the status of the optimizer
        if self.isGur:
            stat = self.m.m.Status
            if stat in [2,13]: return 'optimal'
            elif stat in [3,4]: return 'infeasible'
            elif stat in [5]: return 'unbounded'
            elif stat in [12]: return 'solver error'
            else: return 'other'
        else:
            stat = self.m.m.getsolsta(mosek.soltype.itr)
            if stat in [mosek.solsta.optimal, mosek.solsta.near_optimal]: return 'optimal'
            elif stat in [mosek.solsta.prim_infeas_cer,mosek.solsta.near_prim_infeas_cer]: return 'infeasible'
            elif stat in [mosek.solsta.dual_infeas_cer,mosek.solsta.near_dual_infeas_cer]: return 'unbounded'
            else: return 'other'
            
    def getRHS(self):
        # Returns vector of right-hand-sides (ONLY WORKS FOR MOSEK)
        return self.m.getRHS()
    
    def getZCon(self, rsp):
        # Returns the coefficients of the mean constraint
        rsp = rsp.astype(bool)
        return (2*self.rsp-1)*(np.mean(self.K[rsp],axis=0)-np.mean(self.K[~rsp],axis=0)) if self.dual\
        else np.mean(self.m.dat[rsp],axis=0)-np.mean(self.m.dat[~rsp],axis=0)
    
    def getK(self,test):
        # Returns the Grammian K(X,test) (ONLY FOR KERNEL SVM)
        if len(test.shape)==1: return np.array([self.kernel(x1,test) for x1 in self.m.dat]).reshape((self.numPoints,1))
        return np.array([self.kernel(x1,x2) for x1 in self.m.dat for x2 in test]).reshape((self.numPoints,test.shape[0] if len(test.shape)>1 else 1))
    
    def getSig(self,rsp):
        # For some subset of the data, returns the mean-normalized covariance matrix
        mat = self.K if self.dual else self.m.dat
        return mat[rsp].T.dot(np.eye(sum(rsp))-np.ones((sum(rsp),sum(rsp)))/sum(rsp)).dot(mat[rsp])/sum(rsp)
        #return MinCovDet().fit(mat[rsp]+np.random.normal(scale=1e-6,size=mat[rsp].shape)).covariance_
    
    def pred(self,test):
        # Given the results of the model object, generates predictions on some test set
        # (NOT AVAILABLE FOR GUROBI)
        if self.isGur:
            print('Functionality not available with Gurobi')
            return None
        if self.dual:
            return self.getK(test).T.dot(self.alpha[:,0]*(2*self.rsp-1))
        else:
            return test.dot(self.B)[:,0]
            
    def setLam(self, lam) -> None:
        # Fixes hyperparameter lambda
        self.m.setLam(lam)
        
    def addConstr(self, coeff, idx=0, rhs=0, label=None, record=True) -> None:
        # Handles the addition of a single linear constraint and records it
        self.m.addConstr(coeff,idx,rhs,label,record) if self.isGur\
        else self.m.addConstr(np.tensordot(coeff,coeff,axes=0),rhs**2)
        #else self.m.addConstr(coeff, rhs, record) CHANGE BACK IF NOT MOSPCAMULT
        if record: self.numProjCons += 1
    
    def addQuadConstr(self, rsp, mu=1, B0=None, dualize=True):
        # Handles the addition of a single covariance constraint (ONLY FOR MOSEK)
        if self.isGur:
            print('Functionality not available with Gurobi')
            return None
        if self.isPCA:
            self.m.addQuadConstr(self.getSig(rsp)-self.getSig(~rsp), mu, dualize)
        else:
            eigs, V = la.eigh(self.getSig(rsp)-self.getSig(~rsp))
            pos = (eigs>0)
            V1, V2 = V[:,pos].dot(np.diag(np.sqrt(eigs[pos]))), V[:,~pos].dot(np.diag(np.sqrt(-eigs[~pos])))
            if self.dual:
                V1 *= (2*self.rsp-1).reshape((len(rsp),1)).dot(np.ones((1,V1.shape[1])))
                V2 *= (2*self.rsp-1).reshape((len(rsp),1)).dot(np.ones((1,V2.shape[1])))
            self.V1 = V1
            self.V2 = V2
            self.U1 = V1.dot(V1.T)
            self.U2 = V2.dot(V2.T)
            return self.m.addQuadConstr(V1, V2, mu, B0)
    
    def relaxConstr(self, rhs) -> None:
        # Relaxes a set of constraints (ONLY FOR MOSEK)
        self.m.relaxConstr(rhs)
    
    def updateQuadConstr(self, B0) -> None:
        # Updates the linear portion of a relaxed covariance constraint
        # (ONLY FOR CONVEX-CONCAVE PROCEDURE IN SVM AND WITH MOSEK)
        if self.isGur:
            print('Functionality not available with Gurobi')
        else:
            self.m.updateQuadConstr(B0, self.U1, self.U2)
            
    def projCons(self, projMat=None) -> None:
        # Projects all data according to given matrix and updates optimization model
        # (ONLY FOR MOSEK)
        if self.isGur:
            print('Functionality not available with Gurobi')
        else:
            self.m.projCons(projMat)
            
    def lambdaCrossVal(self, folds=None, lams=[1e-3,1e-2,1e-1,1,1e1,1e2], errType=0, resp=None, k=5):
        # Given potential lambda values and desired error metrix, runs cross-validation
        # via splitting of problem data into training and testing sets, and sets lambda
        # to the value that returns the best average results
        if len(lams)==1:
            self.m.setLam(lams[0])
            return lams[0]
        if folds is None: folds = self.kFold(k)
        rsp = self.rsp if resp is None else resp
        avgErrs = []
        for lam in lams:
            avgErr = 0
            self.m.setLam(lam)
            for fold in folds:
                dat1 = self.m.dat[fold]
                rsp1 = rsp[fold]
                self.m.nullifyConstrs(fold)
                self.optimize()
                if self.getStatus()=='optimal':
                    if errType==0:
                        avgErr += maxROC(self,dat1,rsp1)[0]
                    elif errType==1:
                        avgErr += abs(np.sum(rsp1*(dat1.dot(self.B)+self.b>=0)[:,0])/(np.sum(rsp1))\
                                      -np.sum((1-rsp1)*(dat1.dot(self.B)+self.b>=0)[:,0])/np.sum(1-rsp1))
                    elif errType==2:
                        avgErr += (self.ObjVal-lam*self.B.T.dot(self.B))
                    else:
                        print('Invalid error type, using max ROC error')
                        avgErr += maxROC(self,self.m.dat,self.rsp)[0]
                else:
                    avgErr += 1e12
                self.m.reinstateConstrs(fold) if self.isGur else self.m.reinstateConstrs()
            avgErrs.append(avgErr)
        #if plot:
        #    plt.plot([math.log(lam) for lam in lams],avgErrs)
        optLam = lams[avgErrs.index(max(avgErrs))]
        self.m.setLam(optLam)
        #print('Optimal lambda:',np.log10(optLam))
        return optLam