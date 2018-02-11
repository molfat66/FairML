# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 19:01:17 2017

@author: mahbo

Mosek implementation of fair SVM, with functionality for both kernal and linear SVM.
Functionality also provided for iterative fair SVM procedure, although this is not used

"""

import mosek, math, copy, time, pdb, numpy as np, scipy.linalg as la

def streamprinter(text):
    sys.stdout.write(text)
    sys.stdout.flush()

class mosSVM():
    
    def __init__(self, rsp, dat, lam=1, conic=True, dual=False, K=None, Y=None, presolveTol=1.0e-30, outputFlag=False):
        numPnt = dat.shape[0]
        self.K = K
        self.Y = Y
        self.numFields = dat.shape[1]
        self.numPoints = numPnt
        self.dat = dat
        # To normalize variances
        #for col,var in enumerate(np.var(self.dat,axis=0)):
        #    self.dat[:,col] -= np.mean(self.dat[:,col])
        #    self.dat[:,col] *= 1/math.sqrt(var)
        #    self.dat[:,col] += np.mean(self.dat[:,col])
        self.rsp = rsp
        self.lam = lam
        self.dual = dual
        self.alpha = [0.]*self.numPoints
        self.B = [0.]*self.numFields
        self.b = 0.0
        self.eps = [0.]*self.numPoints
        self.RunTime = 0
        self.conList = []
        self.quadConstr = False
        self.conic = conic
        
        # Make mosek environment
        with mosek.Env() as env:
        
            # Create a task object and attach log stream printer
            with env.Task(0,0) as task:
                task.putdouparam(mosek.dparam.presolve_tol_x,presolveTol)
                # options for convexity check are none, simple, full
                task.putintparam(mosek.iparam.check_convexity,mosek.checkconvexitytype.none)
                if outputFlag: task.set_Stream(mosek.streamtype.msg, streamprinter)
        
                if not dual:
                    # Bound keys for constraints, vars
                    bkc = [mosek.boundkey.lo]*numPnt
                    bkx = [mosek.boundkey.fr]*(self.numFields+1) + [mosek.boundkey.lo]*numPnt
                    blc = [1.0]*numPnt
                    buc = [+math.inf]*numPnt
                    blx = [-math.inf]*(self.numFields+1) + [0.0]*numPnt
                    bux = [+math.inf]*(self.numFields+numPnt+1)
            
                    # Below is the sparse representation of the A
                    # matrix stored by row.
                    asub = [list(range(self.numFields+1))+[self.numFields+i+1] for i in range(numPnt)]
                    aval = [list((2*rsp[i]-1)*dat[i])+[2*rsp[i]-1.0,1.0] for i in range(numPnt)]
            
                    numvar = self.numFields + numPnt + 1
                    numcon = len(bkc)
            
                    task.appendvars(numvar)
                    task.appendcons(numcon)
                    
                    # Set objective.
                    task.putclist([self.numFields+i+1 for i in range(numPnt)], [lam]*numPnt)
            
                    task.putvarboundslice(0, numvar, bkx, blx, bux)
                    task.putconboundslice(0, numcon, bkc, blc, buc)
                    for i in range(numcon):
                        task.putarow(i, asub[i], aval[i])
                        
                    if self.conic:
                        # t>=0, z==1
                        task.appendvars(2)
                        task.putvarbound(task.getnumvar()-2,mosek.boundkey.up,0.0,math.inf)
                        task.putvarbound(task.getnumvar()-1,mosek.boundkey.fx,1.,1.)
                        # t*z>=sumsqr(B), obj coeff of t == 1
                        task.appendcone(mosek.conetype.rquad,0.0,[task.getnumvar()-2,task.getnumvar()-1]+list(range(self.numFields)))
                        task.putcj(task.getnumvar()-2,1.)
                    else:
                        # sumqrt(B) in obj
                        qsubi = list(range(self.numFields))
                        qsubj = list(range(self.numFields))
                        qval = [1.0]*self.numFields
                        task.putqobj(qsubi,qsubj,qval)
                
                else:                    
                    # Bound keys for constraints, vars
                    bkc = [mosek.boundkey.fx]
                    bkx = [mosek.boundkey.ra]*numPnt
                    blc = [0.0]
                    buc = [0.0]
                    blx = [0.0]*numPnt
                    bux = [lam]*numPnt
            
                    numvar = numPnt
                    numcon = 1
            
                    task.appendvars(numvar)
                    task.appendcons(numcon)
                    
                    # Set objective.
                    task.putclist(list(range(numPnt)), [-1.0]*numPnt)
            
                    task.putvarboundslice(0, numvar, bkx, blx, bux)
                    task.putconboundslice(0, numcon, bkc, blc, buc)
                    task.putarow(0, range(numPnt), 2*rsp-1)
                    
                    qval = K*Y+1e-6*np.eye(len(K))
                    if self.conic:
                        self.QChol = la.cholesky(qval,lower=True) # such that K*Y = QChol.dot(QChol.T)
                        task.appendvars(self.QChol.shape[1]+2)
                        task.appendcons(self.QChol.shape[1])
                        # c unbounded, t>=0, z==1
                        task.putvarboundslice(numPnt,numPnt+self.QChol.shape[1],[mosek.boundkey.fr]*self.QChol.shape[1],[-math.inf]*self.QChol.shape[1],[math.inf]*self.QChol.shape[1])
                        task.putvarbound(numPnt+self.QChol.shape[1],mosek.boundkey.lo,0.0,math.inf)
                        task.putvarbound(numPnt+self.QChol.shape[1]+1,mosek.boundkey.fx,1.0,1.0)
                        task.putconboundslice(1,self.QChol.shape[1]+1,[mosek.boundkey.fx]*self.QChol.shape[1],[0.]*self.QChol.shape[1],[0.]*self.QChol.shape[1])
                        i,j = np.meshgrid(range(1,self.QChol.shape[1]+1),range(numPnt))
                        # QChol.T.dot(alpha)==c
                        task.putaijlist(i.flatten(),j.flatten(),self.QChol.flatten())
                        task.putaijlist(range(1,self.QChol.shape[1]+1),range(numPnt,numPnt+self.QChol.shape[1]),[-1.]*self.QChol.shape[1])
                        # t*z>=sumsqr(c)==alpha.T.dot(K*Y).dot(alpha)
                        task.appendcone(mosek.conetype.rquad,0.0,[task.getnumvar()-2,task.getnumvar()-1]+list(range(numPnt,numPnt+self.QChol.shape[1])))
                        task.putcj(task.getnumvar()-2,1.)
                    else:
                        qsubj, qsubi = np.meshgrid(range(numPnt),range(numPnt))
                        idx = np.where(np.tril(np.ones_like(qval)).flatten()!=0)
                        task.putqobj(qsubi.flatten()[idx],qsubj.flatten()[idx],qval.flatten()[idx])
                
                task.putobjsense(mosek.objsense.minimize)
                self.m = task
                
    def optimize(self) -> None:
        runTime = time.time()
        self.m.optimize()
        self.RunTime = time.time()-runTime
        if self.dual:
            xx = [0.]*self.m.getnumvar()
            self.m.getxx(mosek.soltype.itr, xx)
            self.alpha = copy.deepcopy(xx)
            if self.quadConstr:
                self.t = copy.deepcopy(xx[-1])
            self.alpha = copy.deepcopy(self.alpha[:self.numPoints])
            self.B = sum([a*(2*y-1)*row for (a,y,row) in zip(self.alpha,self.rsp,self.dat)])
            self.b = np.mean([2*self.rsp[i]-1-self.K[i].dot(self.alpha*(2*self.rsp-1))\
                              for i in range(self.numPoints) if self.alpha[i]>0]) if max(self.alpha)>0 else 0.
        else:
            xx = [0.]*self.m.getnumvar()
            self.m.getxx(mosek.soltype.itr, xx)
            self.B = copy.deepcopy(xx[:self.numFields])
            self.b = copy.deepcopy(xx[self.numFields])
            self.eps = copy.deepcopy(xx[self.numFields+1:])
            if self.quadConstr:
                self.t = copy.deepcopy(xx[-1])
            self.eps = copy.deepcopy(self.eps[:self.numPoints])
            self.alpha = [self.lam if e>1e-8 else 0 for e in self.eps]
    
    def getAlpha(self):
        return np.array(self.alpha)
    
    def getB(self):
        return np.array(self.B)
    
    def getb(self):
        return self.b
    
    def getObj(self):
        return self.m.getsolutioninfo(mosek.soltype.itr)[0]
    
    def getRHS(self):
        if self.conic: print('WARNING: currently using conic model, results may be inaccurate')
        numcons = self.m.getnumcon()-self.numPoints
        bk = [mosek.boundkey.lo]*numcons
        bl = [0.]*numcons
        bu = [0.]*numcons
        self.m.getconboundslice(self.numPoints,self.m.getnumcon(),bk,bl,bu)
        return np.array(bu)
    
    def setLam(self,lam) -> None:
        if not self.dual: self.m.putclist([self.numFields+i+1 for i in range(self.numPoints)], [lam]*self.numPoints)
        else: self.m.putvarboundslice(0, self.numPoints, [mosek.boundkey.ra]*self.numPoints, [0.]*self.numPoints, [lam]*self.numPoints)
        
    def addConstr(self, coeff, rhs=0, record=True) -> None:
        self.m.appendcons(1)
        self.m.putconbound(self.m.getnumcon()-1, mosek.boundkey.fx if rhs==0 else mosek.boundkey.ra, -rhs, rhs)
        self.m.putarow(self.m.getnumcon()-1, range(len(coeff)), coeff)
        if record: self.conList.append(self.m.getnumcon()-1)
        
    def addQuadConstr(self, V1, V2, mu=1, B0=None):
        U1 = V1.dot(V1.T)
        U2 = V2.dot(V2.T)
        if B0 is None:
            self.optimize()
            B0 = np.array(self.alpha if self.dual else self.B).\
            reshape((len(self.alpha if self.dual else self.B),1))
        numAdd = self.numPoints if self.dual else self.numFields
        if self.conic:
            varOffset = self.m.getnumvar()
            conOffset = self.m.getnumcon()
            # numAdd = V1.shape[1]+V2.shape[1]    
            # dim(c1)==V1.shape[1], dim(c2)==V2.shape[1], dim(t1,t2,z1,z2,t)==1
            self.m.appendvars(numAdd + 5)
            self.m.appendcons(numAdd + 2)
            numvar = self.m.getnumvar()
            numcon = self.m.getnumcon()
            # c1,c2 unbounded, t1,t2>= 0
            self.m.putvarboundslice(varOffset, varOffset + numAdd + 2, [mosek.boundkey.fr]*numAdd+[mosek.boundkey.lo]*2, [-math.inf]*numAdd+[0.]*2, [math.inf]*(numAdd+2))
            self.m.putconboundslice(conOffset, conOffset + numAdd, [mosek.boundkey.fx]*numAdd, [0.]*numAdd, [0.]*numAdd)
            self.m.putconboundslice(conOffset+numAdd, conOffset+numAdd+2, [mosek.boundkey.fx]*2, [-B0.T.dot(U2).dot(B0),-B0.T.dot(U1).dot(B0)], [-B0.T.dot(U2).dot(B0),-B0.T.dot(U1).dot(B0)])
            # z1==1, z2==1, t unbounded
            self.m.putvarboundslice(varOffset + numAdd + 2, varOffset + numAdd + 4, [mosek.boundkey.fx]*2, [1.]*2, [1.]*2)
            self.m.putvarbound(numvar-1,mosek.boundkey.fr,-math.inf,math.inf)
            # V1*B==c1, V2*B==c2 OR V1*alpha==c1, V2*alpha==c2
            i,j = np.meshgrid(range(conOffset,conOffset+numAdd),range(numAdd))
            self.m.putaijlist(i.flatten(),j.flatten(),np.hstack((V1,V2)).flatten())
            self.m.putaijlist(range(conOffset,conOffset+numAdd),range(varOffset,varOffset+numAdd),[-1.]*numAdd)
            # 2*B0*U2*B-B0*U2*B0+t==t1, 2*B0*U1*B-B0*U1*B0+t==t2 OR 2*alpha0*U2*alpha-alpha0*U2*alpha0+t==t1, 2*alpha0*U1*alpha-alpha0*U1*alpha0+t==t2
            self.m.putarow(numcon-2,list(range(numAdd))+[numvar-5,numvar-1],list(-2*U2.dot(B0).flatten())+[1.,-1.])
            self.m.putarow(numcon-1,list(range(numAdd))+[numvar-4,numvar-1],list(-2*U1.dot(B0).flatten())+[1.,-1.])
            # t1*z1>=sumsqr(c1), t2*z2>=sumsqr(c2)
            self.m.appendcone(mosek.conetype.rquad,0.0,[numvar-5,numvar-3]+list(range(varOffset,varOffset+V1.shape[1])))
            self.m.appendcone(mosek.conetype.rquad,0.0,[numvar-4,numvar-2]+list(range(varOffset+V1.shape[1],varOffset+numAdd)))
        else:
            self.m.appendvars(1)
            self.m.appendcons(2)
            numvar = self.m.getnumvar()
            numcon = self.m.getnumcon()
            i,j = np.meshgrid(range(len(U1)),range(len(U1)))
            i = i.flatten(); j = j.flatten()
            idx = np.where(np.tril(np.ones_like(U1)).flatten()!=0)
            self.m.putvarbound(numvar-1,mosek.boundkey.fr,-math.inf,math.inf)
            # B*U1*B-2*B0*U2*B-t<=-B0*U2*B0, B*U2*B-2*B0*U1*B-t<=-B0*U1*B0
            self.m.putqconk(numcon-2,j[idx],i[idx],U1.flatten()[idx])
            self.m.putqconk(numcon-1,j[idx],i[idx],U2.flatten()[idx])
            self.m.putconbound(numcon-2,mosek.boundkey.up,-math.inf,-B0.T.dot(U2).dot(B0))
            self.m.putconbound(numcon-1,mosek.boundkey.up,-math.inf,-B0.T.dot(U1).dot(B0))
            self.m.putarow(numcon-2,list(range(numAdd))+[numvar-1],list(-2*U2.dot(B0).flatten())+[-1.])
            self.m.putarow(numcon-1,list(range(numAdd))+[numvar-1],list(-2*U1.dot(B0).flatten())+[-1.])
        self.m.putcj(self.m.getnumvar()-1,mu)
        self.quadConstr = True
        return B0
        
    def updateQuadConstr(self, B0, U1, U2) -> None:
        numAdd = self.numPoints if self.dual else self.numFields
        numvar = self.m.getnumvar()
        numcon = self.m.getnumcon()
        self.m.putconbound(self.m.getnumcon()-2,mosek.boundkey.fx if self.conic else mosek.boundkey.up,-math.inf,-B0.T.dot(U2).dot(B0))
        self.m.putconbound(self.m.getnumcon()-1,mosek.boundkey.fx if self.conic else mosek.boundkey.up,-math.inf,-B0.T.dot(U1).dot(B0))
        self.m.putarow(numcon-2,list(range(numAdd))+[numvar-1],list(-2*U2.dot(B0).flatten())+[-1.])
        self.m.putarow(numcon-1,list(range(numAdd))+[numvar-1],list(-2*U1.dot(B0).flatten())+[-1.])
        if self.conic:
            self.m.putaij(numcon-2,numvar-5,1.)
            self.m.putaij(numcon-1,numvar-4,1.)
    
    def relaxConstr(self, rhs) -> None:
        if self.conic: print('WARNING: currently using conic model, results may be inaccurate')
        if type(rhs) in [float,int]: rhs = np.array([rhs]*len(self.conList))
        if len(rhs)!=len(self.conList):
            print('Constraints and RHS\'s uneven, can\'t relax constraints')
            return
        self.m.putconboundslice(self.numPoints, self.m.getnumcon(),\
                                [mosek.boundkey.ra]*len(self.conList), -rhs, rhs)
    
    def projCons(self, projMat) -> None:
        if self.conic: print('WARNING: currently using conic model, results may be inaccurate')
        if projMat is None: projMat = np.eye(self.numFields)-self.B.dot(self.B.T)/self.B.T.dot(self.B)
        self.dat = copy.deepcopy(self.dat.dot(projMat))
        asub = [list(range(self.numFields+1))+[self.numFields+i+1] for i in range(self.numPoints)]
        aval = [list((2*self.rsp[i]-1)*self.dat[i])+[2*self.rsp[i]-1.0,1.0] for i in range(self.numPoints)]
        [self.m.putarow(i, asub[i], aval[i]) for i in range(self.numPoints)]
        
    def nullifyConstrs(self,fold) -> None:
        if self.dual: [self.m.putvarbound(variable,mosek.boundkey.fx,0.,0.) for variable in fold]
        else: [self.m.putconbound(constraint,mosek.boundkey.fr,-math.inf,+math.inf) for constraint in fold]
                
    def reinstateConstrs(self) -> None:
        if self.dual: self.m.putvarboundslice(0,self.numPoints,[mosek.boundkey.ra]*self.numPoints,[0.]*self.numPoints,[self.lam]*self.numPoints)
        else: self.m.putconboundslice(0,self.numPoints,[mosek.boundkey.lo]*self.numPoints,[1.0]*self.numPoints,[+math.inf]*self.numPoints)
        