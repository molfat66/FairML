# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 20:45:15 2017

@author: mahbo
"""

import sys, mosek, math, copy, time, numpy as np

def streamprinter(text):
    sys.stdout.write(text)
    sys.stdout.flush()

class mosPCA():
    
    def __init__(self, dat, k1=None, k2=None, presolveTol=1.0e-30, outputFlag=False):
        self.numPoints = dat.shape[0]
        self.numFields = dat.shape[1]
        self.dat = dat - np.ones((self.numPoints,1)).dot(sum(dat)[None,:])
        self.B = [0.]*self.numFields
        self.RunTime = 0
        self.conList = []
        self.quadConstr = False
        self.sparse = False
        
        # Make mosek environment
        with mosek.Env() as env:
        
            # Create a task object and attach log stream printer
            with env.Task(0,0) as task:
                task.putdouparam(mosek.dparam.presolve_tol_x,presolveTol)
                # options for convexity check are none, simple, full
                task.putintparam(mosek.iparam.check_convexity,mosek.checkconvexitytype.none)
                if outputFlag: task.set_Stream(mosek.streamtype.msg, streamprinter)
                    
                # Bound keys for constraints, vars
                barvardim = self.numFields
                numvar = int(barvardim*(barvardim+1)/2) if self.sparse else 0
                numcon = 1 + (barvardim*(barvardim+1) + 1 if self.sparse else 0)
                
                bkc = [mosek.boundkey.fx] + ([mosek.boundkey.lo]*(barvardim*(barvardim+1)) + [mosek.boundkey.lo] if self.sparse else [])
                blc = [1.0] + ([0.0]*(barvardim*(barvardim+1)) + [k1] if self.sparse else [])
                buc = [1.0] + ([0.0]*(barvardim*(barvardim+1)) + [math.inf] if self.sparse else [])
                bkx = [mosek.boundkey.lo]*numvar if self.sparse else []
                blx = [0.0]*numvar if self.sparse else []
                bux = [math.inf]*numvar if self.sparse else []
                
                task.appendvars(numvar)
                task.appendcons(numcon)
                if self.sparse: task.putvarboundslice(0,numvar,bkx,blx,bux)
                task.putconboundslice(0, numcon, bkc, blc, buc)
                task.appendbarvars([barvardim])
                
                i,j = np.tril_indices(barvardim)
                task.putbarcj(0,[task.appendsparsesymmat(barvardim,i,j,self.dat.T.dot(self.dat)[np.tril_indices(self.numFields)])],[1.0])
                task.putbaraij(0,0,[task.appendsparsesymmat(barvardim,range(self.numFields),range(self.numFields),[1.0]*self.numFields)],[1.0])
                if self.sparse:
                    k=1
                    for i in range(self.numFields):
                        for j in range(i):
                            task.putbaraij(k,0,[task.appendsparsesymmat(barvardim,i,j,1.0)],[1.0])
                            task.putbaraij(k+int(barvardim*(barvardim+1)/2),0,[task.appendsparsesymmat(barvardim,i,j,-1.0)],[1.0])
                            k += 1
                    # FIGURE OUT IF MULTIPLIES MATRICES BY 0.5
                    task.putaijlist(range(barvardim*(barvardim+1)),list(range(numvar))*2,[-1.0]*(2*numvar))
                    task.putarow(numcon-1,range(numvar),[1.0]*numvar)
                    
                task.putobjsense(mosek.objsense.maximize)
                self.m = task
                
    def optimize(self) -> None:
        runTime = time.time()
        self.m.optimize()
        self.RunTime = time.time()-runTime
        barvardim = self.numFields
        barx = [0.]*int(barvardim*(barvardim+1)/2)
        self.m.getbarxj(mosek.soltype.itr, 0, barx)
        X = np.zeros((barvardim,barvardim))
        X[np.triu_indices(barvardim)] = barx
        self.X = X + np.triu(X,1).T
        x = np.diag(X)
        if self.m.getnumvar()>0:
            xx = [0.]*self.m.getnumvar()
            self.m.getxx(mosek.soltype.itr,xx)
        self.B = np.sqrt(x[:self.numFields])
        if self.quadConstr:
            self.t = copy.deepcopy(xx[-1])
        
    def getB(self):
        return self.B
    
    def getObj(self):
        return self.m.getsolutioninfo(mosek.soltype.itr)[0]
        
    def addConstr(self, coeff, rhs=0, record=True) -> None:
        if len(coeff)!=self.numFields:
            print('Cannot match coefficient vector of length',len(coeff))
            return
        barvardim = self.numFields
        i,j = np.tril_indices(self.numFields)
        self.m.appendcons(1)
        self.m.putconbound(self.m.getnumcon()-1, mosek.boundkey.fx if rhs==0 else mosek.boundkey.up, 0, rhs**2)
        self.m.putbaraij(self.m.getnumcon()-1,0,[self.m.appendsparsesymmat(barvardim,i,j,coeff[i]*coeff[j])],[1.0])
        if record: self.conList.append(self.m.getnumcon()-1)
        
    def addQuadConstr(self, S, mu=100):
        barvardim = self.numFields
        self.m.appendvars(1)
        self.m.appendcons(2)
        i,j = np.tril_indices(self.numFields)
        self.m.putvarbound(self.m.getnumvar()-1,mosek.boundkey.fr,-math.inf,math.inf)
        self.m.putconboundslice(self.m.getnumcon()-2,self.m.getnumcon(),[mosek.boundkey.up]*2,[-math.inf]*2,[0.]*2)
        self.m.putbaraij(self.m.getnumcon()-2,0,[self.m.appendsparsesymmat(barvardim,i,j,S[np.tril_indices(self.numFields)])],[1.0])
        self.m.putbaraij(self.m.getnumcon()-1,0,[self.m.appendsparsesymmat(barvardim,i,j,-S[np.tril_indices(self.numFields)])],[1.0])
        self.m.putaijlist([self.m.getnumcon()-2,self.m.getnumcon()-1],[self.m.getnumvar()-1]*2,[-1.0]*2)
        self.m.putcj(self.m.getnumvar()-1,-mu)
        
    def projCons(self, projMat=None) -> None:
        if projMat is None: projMat = np.eye(self.numFields)-self.B.dot(self.B.T)/self.B.T.dot(self.B)
        barvardim = self.numFields
        self.dat = copy.deepcopy(self.dat.dot(projMat))
        i,j = np.tril_indices(barvardim)
        self.m.putbarcj(0,[self.m.appendsparsesymmat(barvardim,i,j,self.dat.T.dot(self.dat)[np.tril_indices(self.numFields)])],[1.0])