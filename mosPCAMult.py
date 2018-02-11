# -*- coding: utf-8 -*-
"""
Created on Mon Jan  1 22:39:15 2018

@author: mahbo

Mosek implementation of FPCA that handles all desired PC's at once. Records all
constraints and symmetric matrices defined in the task object for aiding in debugging

"""

import sys, mosek, math, copy, time, numpy as np, scipy.linalg as la

def streamprinter(text):
    sys.stdout.write(text)
    sys.stdout.flush()
    
def getSymMat(mod,name):
    # Given the name of a symmetric matrix, returns its contents in sparse format
    idx = [i for i,(word,nnz) in enumerate(mod.symmats) if word==name]
    if len(idx)==0:
        print('no such symmat')
        return
    if len(idx)>1:
        print('more than one such symmat')
        return
    idx = idx[0]
    i = [0]*mod.symmats[idx][1]
    j = [0]*mod.symmats[idx][1]
    vals = [0.]*mod.symmats[idx][1]
    mod.m.getsparsesymmat(idx,i,j,vals)
    ij = [(i,j) for i,j in zip(i,j)]
    return idx, ij, vals

class mosPCAMult():
    
    def __init__(self, dat, dimPCA, presolveTol=1.0e-30, outputFlag=False):
        self.numPoints = dat.shape[0]
        self.numFields = dat.shape[1]
        self.dimPCA = dimPCA
        self.dat = (np.eye(self.numPoints)-np.ones((self.numPoints,self.numPoints))/self.numPoints).dot(dat)#dat - np.ones((self.numPoints,1)).dot(sum(dat)[None,:])
        for col,var in enumerate(np.var(self.dat,axis=0)):
            self.dat[:,col] *= 1/(math.sqrt(var) if var>0 else 1)
        self.normalizer = la.norm(self.dat)
        self.B = np.zeros((self.numFields,self.dimPCA))
        self.RunTime = 0
        self.normConstr = 0
        self.cons = []
        self.symmats = []
        
        # Make mosek environment
        with mosek.Env() as env:
        
            # Create a task object and attach log stream printer
            with env.Task(0,0) as task:
                task.putdouparam(mosek.dparam.presolve_tol_x,presolveTol)
                # options for convexity check are none, simple, full
                task.putintparam(mosek.iparam.check_convexity,mosek.checkconvexitytype.none)
                task.putintparam(mosek.iparam.infeas_report_auto,mosek.onoffkey.on)
                task.putintparam(mosek.iparam.infeas_report_level,40)
                #task.putdouparam(mosek.dparam.intpnt_co_tol_infeas,1e-5)
                if outputFlag: task.set_Stream(mosek.streamtype.msg, streamprinter)
                    
                # Bound keys for constraints, vars
                barvardim = [self.numFields, self.numFields]
                numvar = 0
                numcon = 1 + int((self.numFields+1)*self.numFields/2)
                idx,jdx = np.tril_indices(self.numFields)
                
                bkc = [mosek.boundkey.fx] + [mosek.boundkey.fx]*(numcon-1)
                blc = [self.dimPCA] + list((idx==jdx).astype(int))
                buc = [self.dimPCA] + list((idx==jdx).astype(int))
                
                task.appendvars(numvar)
                task.appendcons(numcon)
                task.putconboundslice(0, numcon, bkc, blc, buc)
                task.appendbarvars(barvardim)
                
                task.putbarcj(0,[task.appendsparsesymmat(self.numFields,idx,jdx,-(1/self.numPoints/self.normalizer)*self.dat.T.dot(self.dat)[idx,jdx])],[1.0])
                self.symmats.append(('Obj',self.numFields))
                task.putbaraij(0,0,[task.appendsparsesymmat(self.numFields,range(self.numFields),range(self.numFields),[1.0]*self.numFields)],[1.0])
                self.cons.append('traceCon')
                self.symmats.append(('traceCon',self.numFields))
                for count,(i,j) in enumerate(zip(idx,jdx)):
                    task.putbaraij(count+1,0,[task.appendsparsesymmat(self.numFields,[i],[j],[1.0])],[1.0])
                    task.putbaraij(count+1,1,[task.appendsparsesymmat(self.numFields,[i],[j],[1.0])],[1.0])
                    self.cons.append('X<=ICon_%s_%s'%(i,j))
                    self.symmats.append(('X<=ICon%s_%s_%s'%(0,i,j),1))
                    self.symmats.append(('X<=ICon%s_%s_%s'%(1,i,j),1))
                task.putobjsense(mosek.objsense.minimize)
                self.m = task
                
    def optimize(self) -> None:
        runTime = time.time()
        self.m.optimize()
        self.RunTime = time.time()-runTime
        barvardim = self.numFields
        barx = [0.]*int(barvardim*(barvardim+1)/2)
        barx1 = [0.]*int(barvardim*(barvardim+1)/2)
        self.m.getbarxj(mosek.soltype.itr, 0, barx)
        self.m.getbarxj(mosek.soltype.itr, 1, barx1)
        X = np.zeros((barvardim,barvardim))
        Y = np.zeros((barvardim,barvardim))
        X[np.triu_indices(barvardim)] = barx
        Y[np.triu_indices(barvardim)] = barx1
        self.XFull = X + np.triu(X,1).T
        self.Y = Y + np.triu(Y,1).T
        self.X = self.XFull[:barvardim,:barvardim]
        self.B = la.eigh(self.X)[1][:,-self.dimPCA:]
        if self.normConstr>=1:
            barx2 = [0.]*(barvardim*(2*barvardim+1))
            self.m.getbarxj(mosek.soltype.itr, 2, barx2)
            Z0 = np.zeros((2*barvardim,2*barvardim))
            Z0[np.triu_indices(2*barvardim)] = barx2
            self.Z0 = Z0 + np.triu(Z0,1).T
        if self.normConstr>=2:
            barx3 = [0.]*(barvardim*(2*barvardim+1))
            self.m.getbarxj(mosek.soltype.itr, 3, barx3)
            Z1 = np.zeros((2*barvardim,2*barvardim))
            Z1[np.triu_indices(2*barvardim)] = barx3
            self.Z1 = Z1 + np.triu(Z1,1).T
        if self.normConstr>=3:
            barx4 = [0.]*(barvardim*(2*barvardim+1))
            self.m.getbarxj(mosek.soltype.itr, 4, barx4)
            Z2 = np.zeros((2*barvardim,2*barvardim))
            Z2[np.triu_indices(2*barvardim)] = barx4
            self.Z2 = Z2 + np.triu(Z2,1).T
        if self.m.getnumvar()>0:
            self.t = [0.]*self.m.getnumvar()
            self.m.getxx(mosek.soltype.itr,self.t)
        
    def getB(self):
        return self.B
    
    def getObj(self):
        return self.m.getsolutioninfo(mosek.soltype.itr)[0]
        
    def addConstr(self, S, d=0, bound=True):
        barvardim = self.numFields
        numbar = self.m.getnumbarvar()
        basecons = self.m.getnumcon()
        newcons = barvardim*(2*barvardim+1)
        self.m.appendvars(1)
        self.m.appendcons(newcons)
        self.m.appendbarvars([2*barvardim])
        self.m.putvarbound(self.m.getnumvar()-1,mosek.boundkey.lo,0,math.inf)
        self.m.putconboundslice(basecons,self.m.getnumcon(),[mosek.boundkey.fx]*newcons,[0.]*newcons,[0.]*newcons)
        idx,jdx = np.meshgrid(range(barvardim),range(barvardim))
        jtemp,itemp = np.meshgrid(range(barvardim),range(barvardim))
        itemp[np.triu_indices(barvardim)] = idx[np.triu_indices(barvardim)]
        jtemp[np.triu_indices(barvardim)] = jdx[np.triu_indices(barvardim)]
        con = 0
        for i,j in zip(idx.flatten(),jdx.flatten()):
            self.m.putbaraij(basecons+con,0,[self.m.appendsparsesymmat(barvardim,itemp[j],jtemp[j],S[i]*((itemp[j]==jtemp[j])+1))],[1.0])
            self.m.putbaraij(basecons+con,numbar,[self.m.appendsparsesymmat(2*barvardim,[barvardim+i],[j],[-1.0])],[1.0])
            con += 1
            self.cons.append('Z%sBottomLeftCon_%s_%s'%(numbar-2,i,j))
            self.symmats.append(('Z%sBottomLeftCon%s_%s_%s'%(numbar-2,0,i,j),barvardim))
            self.symmats.append(('Z%sBottomLeftCon%s_%s_%s'%(numbar-2,1,i,j),1))
        idx,jdx = np.tril_indices(barvardim)
        for i,j in zip(idx,jdx):
            self.m.putbaraij(basecons+con,numbar,[self.m.appendsparsesymmat(2*barvardim,[i],[j],[1.0])],[1.0])
            if i==j: self.m.putaij(basecons+con,self.m.getnumvar()-1,-1.0)
            con += 1
            self.cons.append('Z%sTopLeftCon_%s_%s'%(numbar-2,i,j))
            self.symmats.append(('Z%sTopLeftCon_%s_%s'%(numbar-2,i,j),1))
        for i,j in zip(idx,jdx):
            self.m.putbaraij(basecons+con,numbar,[self.m.appendsparsesymmat(2*barvardim,[barvardim+i],[barvardim+j],[1.0])],[1.0])
            if i==j: self.m.putaij(basecons+con,0,-1.0)
            con += 1
            self.cons.append('Z%sBottomRightCon_%s_%s'%(numbar-2,i,j))
            self.symmats.append(('Z%sBottomRightCon_%s_%s'%(numbar-2,i,j),1))
        if bound:
            self.m.appendcons(1)
            self.m.putconbound(self.m.getnumcon()-1,mosek.boundkey.up,-math.inf,d/self.normalizer**2)
            self.m.putaij(self.m.getnumcon()-1,self.m.getnumvar()-1,1.0)
            self.cons.append('Z%sLinBound'%(numbar-2))
        self.normConstr += 1
        
    def addQuadConstr(self, S, mu=1, dualize=True):
        L,V = la.eigh(S)
        #self.addConstr(np.diag(np.sqrt(L+max(0,-min(L)))).dot(V.T),bound=True)
        #self.addConstr(np.diag(np.sqrt(-L+max(0,max(L)))).dot(V.T),bound=True)
        self.eps = max(np.abs(L)) + 0.5
        self.addQuadConstr_barvar(V.dot(np.diag(1./(L+self.eps))).dot(V.T),bound=False)
        self.addQuadConstr_barvar(V.dot(np.diag(1./(-L+self.eps))).dot(V.T),bound=False)
        if dualize:
            self.m.appendvars(1)
            self.m.appendcons(2)
            self.m.putvarbound(self.m.getnumvar()-1,mosek.boundkey.lo,0,math.inf)
            self.m.putconboundlist([self.m.getnumcon()-2,self.m.getnumcon()-1],[mosek.boundkey.up]*2,[math.inf]*2,[0.0]*2)
            self.m.putaij(self.m.getnumcon()-2,self.m.getnumvar()-3,1.0)
            self.m.putaij(self.m.getnumcon()-1,self.m.getnumvar()-2,1.0)
            self.m.putaij(self.m.getnumcon()-2,self.m.getnumvar()-1,-1.0)
            self.m.putaij(self.m.getnumcon()-1,self.m.getnumvar()-1,-1.0)
            self.m.putcj(self.m.getnumvar()-1,mu)
        else:
            self.m.putvarboundlist([self.m.getnumvar()-2,self.m.getnumvar()-1],[mosek.boundkey.ra]*2,[0.0]*2,[max(np.abs(L))*mu+self.eps]*2)
        
    
    def addQuadConstr_barvar(self, S, mu=1, dualize=True, bound=True):
        barvardim = self.numFields
        numbar = self.m.getnumbarvar()
        basecons = self.m.getnumcon()
        newcons = barvardim**2
        self.m.appendvars(1)
        self.m.appendcons(newcons)
        self.m.appendbarvars([2*barvardim])
        self.m.putvarbound(self.m.getnumvar()-1,mosek.boundkey.lo,0,math.inf)
        self.m.putconboundslice(basecons,self.m.getnumcon(),[mosek.boundkey.fx]*newcons,[0.]*newcons,[0.]*newcons)
        idx,jdx = np.meshgrid(range(barvardim),range(barvardim))
        con = 0
        for i,j in zip(idx.flatten(),jdx.flatten()):
            self.m.putbaraij(basecons+con,0,[self.m.appendsparsesymmat(barvardim,[max(i,j)],[min(i,j)],[2.0 if i==j else 1.0])],[1.0])
            self.m.putbaraij(basecons+con,numbar,[self.m.appendsparsesymmat(2*barvardim,[barvardim+i],[j],[-1.0])],[1.0])
            con += 1
            self.cons.append('Z%sBottomLeftCon_%s_%s'%(numbar-2,i,j))
            self.symmats.append(('Z%sBottomLeftCon%s_%s_%s'%(numbar-2,0,i,j),1))
            self.symmats.append(('Z%sBottomLeftCon%s_%s_%s'%(numbar-2,1,i,j),1))
        idx,jdx = np.tril_indices(barvardim)
        basecons = self.m.getnumcon()
        con = 0
        self.m.appendcons(idx.size)
        self.m.putconboundslice(basecons,self.m.getnumcon(),[mosek.boundkey.fx]*idx.size,[0.]*idx.size,[0.]*idx.size)
        for i,j in zip(idx,jdx):
            self.m.putbaraij(basecons+con,numbar,[self.m.appendsparsesymmat(2*barvardim,[i],[j],[1.0])],[1.0])
            if i==j: self.m.putaij(basecons+con,self.m.getnumvar()-1,-1.0)
            con += 1
            self.cons.append('Z%sTopLeftCon_%s_%s'%(numbar-2,i,j))
            self.symmats.append(('Z%sTopLeftCon_%s_%s'%(numbar-2,i,j),1))
        basecons = self.m.getnumcon()
        self.m.appendcons(idx.size)
        self.m.putconboundslice(basecons,self.m.getnumcon(),[mosek.boundkey.fx]*idx.size,S[idx,jdx],S[idx,jdx])
        con = 0
        for i,j in zip(idx,jdx):
            self.m.putbaraij(basecons+con,numbar,[self.m.appendsparsesymmat(2*barvardim,[barvardim+i],[barvardim+j],[1.0 if i==j else 0.5])],[1.0])
            con += 1
            self.cons.append('Z%sBottomRightCon_%s_%s'%(numbar-2,i,j))
            self.symmats.append(('Z%sBottomRightCon_%s_%s'%(numbar-2,i,j),1))
        if bound:
            if dualize:
                self.m.putcj(self.m.getnumvar()-1,mu)
            else:
                self.m.appendcons(1)
                self.m.putconbound(self.m.getnumcon()-1,mosek.boundkey.up,-math.inf,mu)
                self.m.putaij(self.m.getnumcon()-1,self.m.getnumvar()-1,1.0)
                self.cons.append('Z%sLinBound'%(numbar-2))
        self.normConstr += 1
    
    def projCons(self, projMat=None) -> None:
        if projMat is None: projMat = np.eye(self.numFields)-self.B.dot(self.B.T)/self.B.T.dot(self.B)
        barvardim = self.numFields
        self.dat = copy.deepcopy(self.dat.dot(projMat))
        i,j = np.tril_indices(barvardim)
        self.m.putbarcj(0,[self.m.appendsparsesymmat(barvardim,i,j,self.dat.T.dot(self.dat)[np.tril_indices(self.numFields)])],[1.0])