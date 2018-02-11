# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 17:56:40 2017

@author: mahbo

Gurobi implementation of older versions of fair SVM algorithm (NOT USED)

"""

import copy, numpy as np
from gurobipy import *

class gurMod():
    
    def __init__(self,prob,rsp,dat,lam=1,outputFlag=False):
        self.m = Model('SVM')
        self.m.setParam('outputFlag',outputFlag)
        self.m.ModelSense = GRB.MINIMIZE
        self.coreConstrs = []
    
        self.B = np.array([self.m.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name='{0}_coeff'.format(prob.headers[i])) for i in range(prob.numFields)])
        self.eps = np.array([self.m.addVar(vtype=GRB.CONTINUOUS, name='eps_{0}'.format(i)) for i in range(dat.shape[0])])
        self.b = self.m.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name='b')
    
        for i in range(dat.shape[0]):
            self.coreConstrs.append(self.m.addConstr((2*rsp[i]-1)*(dat[i].dot(self.B)+self.b)>=1-self.eps[i], name='epsConstr{0}'.format(i)))
        
        Q = QuadExpr(self.B.dot(self.B) + lam*np.sum(self.eps))
        self.m.setObjective(Q)
        self.m.update()
        self.m.reset()
        
        self.conList = []
        self.numFields = prob.numFields
        self.numPoints = dat.shape[0]
    
    def optimize(self) -> None:
        self.m.optimize()
        
    def getB(self):
        return copy.deepcopy([x.X for x in self.B])
    
    def getb(self):
        return self.b.X
    
    def getObj(self):
        return self.m.objVal
    
    def getRHS(self):
        return np.array([con[0].rhs for con in self.conList])
    
    def setLam(self, lam) -> None:
        for e in self.eps: e.Obj = lam
        self.m.update()
        self.m.reset()
    
    def addConstr(self, fixBeta, idx=0 , rhs=0, label=None, record=True) -> None:
        L = LinExpr(fixBeta,self.B)
        upCon = self.m.addConstr(L<=rhs, name='betaConstrUp{0}'.format(idx) if label is None else label+'Up')
        dnCon = self.m.addConstr(L>=rhs, name='betaConstrDn{0}'.format(idx) if label is None else label+'Dn')
        self.m.update()
        self.m.reset()
        if record: self.conList.append((upCon,dnCon))
    
    def relaxConstr(self, rhs) -> None:
        if type(rhs) in [float,int]: rhs = [rhs]*len(self.conList)
        if len(rhs)!=len(self.conList):
            print('Constraints and RHS\'s uneven, can\'t relax constraints')
            return
        [(upCon.setAttr('rhs',rhs),dnCon.setAttr('rhs',-rhs)) for ((upCon,dnCon),rhs) in zip(self.conList,rhs)]
        self.m.update()
        self.m.reset()
        
    def nullifyConstrs(self, fold) -> None:
        [constr.setAttr('rhs',-GRB.INFINITY) for constr in fold]
        
    def reinstateConstrs(self, fold) -> None:
        [constr.setAttr('rhs',1) for constr in fold]