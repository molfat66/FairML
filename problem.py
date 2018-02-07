# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 19:42:45 2017

@author: mahbo
"""

import csv, math, winsound, time, copy, cloudpickle, numpy as np, matplotlib.pyplot as plt, scipy.linalg as la
from collections import namedtuple
from scipy.integrate import trapz
from scipy.stats.stats import pearsonr
from scipy.optimize import linear_sum_assignment, newton_krylov
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D
from model import *

kern = namedtuple('kern', 'name xbound ybound main side xthresh ythresh')

def graphROC(main, side, xthresh, ythresh, figsize=(6,6), title=None, axes=True):
        fig = plt.figure(figsize=figsize)
        plt.plot(side[0],side[1],color=[0.8500,0.3250,0.0980])
        #plt.plot(main[0],main[1],color=[0,0.4470,0.7410])
        [[plt.plot([x,x],[0,y],':',color='pink'),plt.plot([0,x],[y,y],':',color='pink')]\
          for (x,y) in zip(xthresh,ythresh)]
        plt.plot([0,1],[0,1],'--',color=[0.5,0.5,0.5])
        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.xticks([0,0.5,1])
        plt.yticks([0,0.5,1])
        if axes: plt.xlabel('False Positives')
        if axes: plt.ylabel('True Positivies')
        if title is not None: plt.title(title)
        for x,y in zip(xthresh,ythresh):
            plt.text(x+0.01,0.01,str(round(x,2)))
            plt.text(0.01,y+0.01,str(round(y,2)))
        return fig

class problem():
    
    def __init__(self, filename=None, numFields=40, numPoints=500, corr=0.8, M=1000, perc=0, isGur=0) -> None:
        self.filename = filename
        self.M = 1000
        if self.filename!=None:
            self.data = []
            with open(self.filename,'r') as simfile:
                filereader = csv.reader(simfile)
                for dPoint in filereader:
                    self.data.append(dPoint)
            self.numPoints = len(self.data)-1
            self.numFields = len(self.data[0])-2
            self.data = np.array([element for dPoint in self.data for element in dPoint]).reshape(self.numPoints+1,self.numFields+2)
            self.headers = self.data[0].copy()
            self.mainResp = self.data[1:,-2].copy().astype(int)
            self.sideResp = self.data[1:,-1].copy().astype(int)
            self.data = self.data[1:,:-2].copy().astype(float)
        else:
            self.numFields = numFields
            self.numPoints = numPoints
            B1 = np.random.random(numFields)
            B1 = B1/la.norm(B1)
            B2 = np.random.random(numFields-1)
            B2 = np.hstack((B2, -B1[:numFields-1].dot(B2)/B1[numFields-1]))
            B2 = B2/la.norm(B2)
            B2 = corr*B1 + math.sqrt(1-corr**2)*B2
            self.headers = ['X'+str(i) for i in range(1,numFields+1)]+['Y1','Y2']
            self.data = 10*(2*np.random.random((numPoints,numFields))-1)
            self.mainResp = np.array([1 if np.random.random()<=math.exp(B1.dot(self.data[i,]))/(1+math.exp(B1.dot(self.data[i,])))\
                                      else 0 for i in range(numPoints)]).astype(int)
            self.sideResp = np.array([1 if np.random.random()<=math.exp(B2.dot(self.data[i,]))/(1+math.exp(B2.dot(self.data[i,])))\
                                      else 0 for i in range(numPoints)]).astype(int)
        self.isSplit = False
        self.isGur = isGur
        if perc>0: self.splitData(perc)
        
    def getMainSideCorr(self):
        return pearsonr(self.mainResp,self.sideResp)[0]
    
    def setGaussData(self, mean1, mean2, cov1, cov2):
        if mean1.size!=mean2.size or cov1.shape!=(mean1.size,mean1.size) or cov2.shape!=(mean1.size,mean1.size):
            print('Incorrect input shapes')
            return
        self.numFields = mean1.size
        B1 = np.random.multivariate_normal(mean1,cov1,math.floor(self.numPoints/2))
        B2 = np.random.multivariate_normal(mean2,cov2,math.ceil(self.numPoints/2))
        self.data = np.vstack((B1,B2))
        self.sideResp = np.array([1]*math.floor(self.numPoints/2)+[0]*math.ceil(self.numPoints/2))
    
    def splitData(self,perc) -> None:
        if perc<0 or perc>1:
            print('Please enter a new percentage to use for training')
            return
        idx = np.arange(self.numPoints)
        np.random.shuffle(idx)
        trainidx = idx[:int(perc*self.numPoints)]
        testidx = idx[int(perc*self.numPoints):]
        self.train = self.data[trainidx]
        self.test = self.data[testidx]
        self.trainMain = self.mainResp[trainidx]
        self.trainSide = self.sideResp[trainidx]
        self.testMain = self.mainResp[testidx]
        self.testSide = self.sideResp[testidx]
        self.numTrain = trainidx.size
        self.numTest = testidx.size
        self.isSplit = True
        
    def checkIfSplit(self, split, test=False):
        if split:
            if not self.isSplit:
                print('Data not split, using standard 70-30 split')
                self.splitData(0.7)
            if test: return self.test, self.testMain, self.testSide, self.numTest
            else: return self.train, self.trainMain, self.trainSide, self.numTrain
        else: return self.data, self.mainResp, self.sideResp, self.numPoints
    
    def pca(self, dimPCA=2, m=None, split=False, outputFlag=False):
        dat, mrsp, srsp, numPnt = self.checkIfSplit(split)
        
        totTime = time.time()
        if m is None: m = model(dat,isPCA=True,dimPCA=dimPCA)
        m.optimize(outputFlag=outputFlag)
        if outputFlag: print("Total running time: %s" % (round(time.time()-totTime,2)))
        return m.B, m
    
    def zpca(self, dimPCA=2, d=0, m=None, split=False, outputFlag=False):
        dat, mrsp, srsp, numPnt = self.checkIfSplit(split)
        
        totTime = time.time()
        if m is None: m = model(dat,isPCA=True,dimPCA=dimPCA,outputFlag=outputFlag)
        m.addConstr(m.getZCon(srsp),rhs=d,label='linCon',record=False)
        m.optimize(outputFlag=outputFlag)
        if outputFlag: print("Total running time: %s" % (round(time.time()-totTime,2)))
        return m.B, m
    
    def spca(self, dimPCA=2, d=0, mu=1, m=None, addLinear=True, dualize=True, split=False, outputFlag=False):
        dat, mrsp, srsp, numPnt = self.checkIfSplit(split)
                
        totTime = time.time()
        optTime = 0
        if m is None: m = model(dat,isPCA=True,dimPCA=dimPCA,outputFlag=outputFlag)
        if addLinear: m.addConstr(m.getZCon(srsp),rhs=d,label='linCon',record=False)
        m.addQuadConstr(srsp.astype(bool),mu,dualize=dualize)
        m.optimize(outputFlag=outputFlag)
        if outputFlag: print("Total running time: %s:%s" % (int((time.time()-totTime)/60),round((time.time()-totTime)%60,2)))
        return m.B, m
    
    def svm(self, m=None, lams=None, conic=True, dual=False, kernel=None, split=False, outputFlag=False, useSRSP=False):
        dat, mrsp, srsp, numPnt = self.checkIfSplit(split,test=useSRSP)

        totTime = time.time()
        if m is None: m = model(dat,mrsp if not useSRSP else srsp,conic=conic,dual=dual,kernel=kernel)
        if lams is not None: lam = m.lambdaCrossVal(lams=lams,errType=0)
        m.optimize()
        err = round(maxROC(m,dat,srsp)[0],4)
        if outputFlag: print("\tOptimization time: %02d:%02d\n\tTotal running time: %02d:%02d\n\tMaximum Error: %s"\
                             % (int(m.RunTime/60),round(m.RunTime%60),int((time.time()-totTime)/60),\
                                round((time.time()-totTime)%60),err))
        return m.B, m, err # may cause errors, remove last item
    
    def zsvm(self, d, m=None, lams=None, conic=True, dual=False, kernel=None, split=False, outputFlag=False):
        dat, mrsp, srsp, numPnt = self.checkIfSplit(split)
        
        totTime = time.time()
        if m is None: m = model(dat,mrsp,conic=conic,dual=dual,kernel=kernel,isGur=self.isGur)
        if lams is not None: lam = m.lambdaCrossVal(lams=lams,errType=2)
        m.addConstr(m.getZCon(srsp),rhs=d,label='linCon',record=False)
        m.optimize()
        if outputFlag: print("Optimization time: %s:%s\nTotal running time: %s:%s\nMaximum Error: %s"\
                             % (int(m.RunTime/60),round(m.RunTime%60),int((time.time()-totTime)/60),\
                                round((time.time()-totTime)%60),round(maxROC(m,dat,srsp)[0],4)))
        return m.B, m
    
    def ssvm(self, d, mu=1, lams=None, conic=True, dual=False, maxiters=10, m=None, B0=None, zeroThresh=1e-6, split=False, outputFlag=False):
        dat, mrsp, srsp, numPnt = self.checkIfSplit(split)
                
        totTime = time.time()
        optTime = 0
        if m is None: m = model(dat,mrsp,conic=conic,dual=dual,isGur=self.isGur)
        
        m.addConstr(m.getZCon(srsp),rhs=d,label='linCon',record=False)
        if lams is not None: lam = m.lambdaCrossVal(lams=lams,errType=2)
        prevB = m.addQuadConstr(srsp.astype(bool),mu,B0)
        optTime += m.RunTime
        
        for i in range(maxiters):
            m.optimize()
            optTime += m.RunTime
            if la.norm(m.alpha if dual else m.B)<zeroThresh:
                print('Norm of beta less than threshold..')
                break
            if la.norm((m.alpha if dual else m.B) - prevB)<zeroThresh:
                break
            else:
                prevB = copy.deepcopy(m.alpha if dual else m.B)
            m.updateQuadConstr(prevB)
            if outputFlag: print('Done iteration',i,m.m.t)
        if outputFlag: print("Optimization time: %s:%s\nTotal running time: %s:%s\nMaximum Error: %s"\
                             % (int(optTime/60),round(optTime%60),int((time.time()-totTime)/60),\
                                round((time.time()-totTime)%60),round(maxROC(m,dat,srsp)[0],4)))
        return m.B, m
    
    def iterRun(self, d=0.1, m=None, maxiters=20, maxRelax=10, zeroThresh=1e-6, binSearch=1,\
                lams=None, split=False, outputFlag=True):
        dat, mrsp, srsp, numPnt = self.checkIfSplit(split)
        optTime = 0
        totTime = time.time()
        print('')
    
        # create models
        sideMod = model(dat,srsp,isGur=self.isGur)
        mainMod = model(dat,mrsp,isGur=self.isGur)
        
        # run unconstrained SVM and plot for comparison
        #if lams is not None: mainMod.lambdaCrossVal(lams=lams,errType=2,resp=srsp)
        #mainMod.optimize()
        #compB = mainMod.B
        
        # vessels to record info
        bMat = np.zeros((self.numFields,self.numFields))
        projMat = np.identity(self.numFields)
        norms, objs = [],[]
        rhsRelax = 0.01
        relaxed = 0
    
        for i in range(min(self.numFields,maxiters)):
            # RUN SVM ON THE SIDE, CHECK IF IT WORKED (RELAX CONSTRAINTS IF BETA IS TOO SMALL), AND RECORD RELEVANT INFO
            if lams is not None: sideLam = sideMod.lambdaCrossVal(lams=lams,errType=2)
            sideMod.optimize()
            optTime += sideMod.RunTime
            # force projections if facing numerical issues
            if sideMod.getStatus()!='optimal':
                print('Unable to solve side model (status %s), projecting data..'%sideMod.getStatus())
                rhsRelax = 0
                sideMod = model(dat.dot(projMat),srsp,lam=sideLam if lams is not None else 1,isGur=self.isGur)
                mainMod = model(dat.dot(projMat),mrsp,isGur=self.isGur)
                sideMod.optimize()
                optTime += sideMod.RunTime
                if sideMod.getStatus()!='optimal':
                    print('No solution for side model (status %s), exiting..'%sideMod.getStatus())
                    break
            # if returned vector too small in magnitude relax constraints slightly to avoid numerical issues
            while la.norm(sideMod.B)<zeroThresh:
                print('Norm of beta less than threshold, relaxing previous constraints on side model.')
                relaxed += 1
                if relaxed > maxRelax: break
                sideMod.relaxConstr(sideMod.getRHS()+rhsRelax)
                mainMod.relaxConstr(mainMod.getRHS()+rhsRelax)
                sideMod.optimize()
                optTime += sideMod.RunTime
            if rhsRelax > maxRelax:
                print('Unable to increase norm of beta via constraint relaxation, exiting..')
                break
            # record info
            norms.append(la.norm(sideMod.B))
            bMat[:,i] = copy.deepcopy(sideMod.B/norms[i])[:,0]
            projMat -= sideMod.B.dot(sideMod.B.T)/norms[i]**2
            objs.append(copy.copy(sideMod.ObjVal))
            
            # ADD CONSTRAINT TO ALL MODELS
            mainMod.addConstr(sideMod.B)
            sideMod.addConstr(sideMod.B)
            
            # RUN UPDATED SVM ON MAIN MODEL, CHECK IF IT WORKED, AND CHECK IF MAX ERROR IS WITHIN TOLERANCE
            if lams is not None: mainLam = mainMod.lambdaCrossVal(lams=lams,errType=2)
            mainMod.optimize()
            optTime += mainMod.RunTime
            # force projections if facing numerical issues
            if mainMod.getStatus()!='optimal':
                print('Unable to solve main model (status %s), projecting data..'%mainMod.getStatus())

                sideMod = model(dat.dot(projMat),srsp,isGur=self.isGur)
                mainMod = model(dat.dot(projMat),mrsp,lam=mainLam if lams is not None else 1,isGur=self.isGur)
                mainMod.optimize()
                optTime += mainMod.RunTime
            if mainMod.getStatus()!='optimal':
                print('No solution for main model (status %s), exiting..'%(mainMod.getStatus()))
                break
            # calculate max unfairness, break if within threshold
            bestB = copy.deepcopy(mainMod.B)
            maxErr = maxROC(mainMod,dat,srsp)[0]
            if outputFlag: print('Iteration %s error: %s' % (i+1,round(maxErr,4)),\
                                 (', side lambda: %s, main lambda: %s'%(sideLam,mainLam))\
                                 if lams is not None else '')
            if maxErr<d: break
        bMat = bMat[:,:i+1]
        
        # conduct binary search to relax constraints to maximize predictability of main variable subject to fairness bounds
        j = -1
        if binSearch: maxErr, bestB, j = self.binarySearch(d,mainMod,dat,srsp,\
                                                           dnBnd=np.zeros(mainMod.numProjCons) if maxErr>=d else None)
        else: bestB = copy.deepcopy(mainMod.B)
        totTime = round(time.time() - totTime,2)
        if outputFlag:
            print('\nTotal running time: %s, Optimization time: %s'%(round(totTime,2),round(optTime,2)))
            print('Ran %s outer iteration(s) and %s binary search iteration(s)'%(i+1,j+1))
            print('Found maximum error of %s'%round(maxErr,4))
        return bestB, bMat, norms, objs, projMat, mainMod
    
    def binarySearch(self, d, m, dat, srsp, upBnd=None, dnBnd=None, thresh=1e-3, maxiters=50):
        if dnBnd is None: dnBnd = m.getRHS()
        if upBnd is None: upBnd = np.ones(m.numProjCons)
        prevMaxErr = maxROC(m,dat,srsp)[0]
        for i in range(maxiters):
            m.relaxConstr((upBnd+dnBnd)/2)
            m.optimize()
            if m.getStatus() in ['infeasible','solver error','other']:
                print('Encountered solver error, status %s, assuming error = -inf'%m.getStatus())
                dnBnd = copy.deepcopy((upBnd+dnBnd)/2)
                continue
            maxErr = maxROC(m,dat,srsp)[0]
            print('Ran iteration %s in binary search, found max error of %s' % (i+1,round(maxErr,4)))
            if maxErr<=d:
                if maxErr+thresh>d or round(maxErr,8)==round(prevMaxErr,8): break
                dnBnd = copy.deepcopy((upBnd+dnBnd)/2)
            #elif abs(prevMaxErr-maxErr)<0.01*thresh: break
            else: upBnd = copy.deepcopy((upBnd+dnBnd)/2)
            prevMaxErr = maxErr
        if maxErr>d:
            m.relaxConstr((upBnd+dnBnd)/2)
            m.optimize()
        return maxROC(m,dat,srsp)[0], m.B, i
    
    def ROC(self, m=None, perc=0.25, pred=None, split=False):
        dat, mrsp, srsp, numPnt = self.checkIfSplit(split, test=True)
        
        if pred is None and m is None: print('Not enough inputs')
        if pred is None: pred = m.pred(dat)
        predSort = np.argsort(pred.flatten()).astype(int)
        mrsp = mrsp[predSort]
        srsp = srsp[predSort]
        msum1 = np.sum(mrsp); msum0 = np.sum(1-mrsp)
        ssum1 = np.sum(srsp); ssum0 = np.sum(1-srsp)
        brange = np.sort(np.random.choice(range(len(predSort)),int(perc*len(predSort)),replace=False))
        main = np.array([[np.sum(1-mrsp[b:])/msum0 for b in brange],[np.sum(mrsp[b:])/msum1 for b in brange]])
        side = np.array([[np.sum(1-srsp[b:])/ssum0 for b in brange],[np.sum(srsp[b:])/ssum1 for b in brange]])
        return main, side
    
    def ROCStats(self, m, perc=0.5, split=False):
        main, side = self.ROC(m,perc,split)
        return trapz(main[1],main[0]), max(np.abs(side[1]-side[0]))
    
    def plot(self, m=None, perc=0.5, pred=None, figsize=(6,6), split=False, thresholds=[], title=None, graph=True):
        dat, mrsp, srsp, numPnt = self.checkIfSplit(split, test=True)
        if pred is None and m is None: print('Not enough inputs')
        if pred is None: pred = m.pred(dat)
        
        main, side = self.ROC(m,perc=perc,pred=pred,split=split)
        xthresh = [np.sum((1-srsp)*(pred>=b))/np.sum(1-srsp) for b in thresholds]
        ythresh = [np.sum(srsp*(pred>=b))/(np.sum(srsp)) for b in thresholds]
        if graph: graphROC(main, side, xthresh, ythresh, figsize, title)
        return main, side, xthresh, ythresh
    
    def plotComp(self, m1, m2, perc=0.25, title=None, figsize=(6,6), split=False, method1='original', method2='modified'):
        dat, mrsp, srsp, numPnt = self.checkIfSplit(split, test=True)
        
        main1, side1 = self.ROC(m1,perc=perc,split=split)
        main2, side2 = self.ROC(m2,perc=perc,split=split)
        plt.figure(figsize = figsize)
        newSide = plt.plot(side2[0],side2[1],color=[0.8500,0.3250,0.0980])
        newMain = plt.plot(main2[0],main2[1],color=[0,0.4470,0.7410])
        oldSide = plt.plot(side1[0],side1[1],':',color=[0.8500,0.3250,0.0980])
        oldMain = plt.plot(main1[0],main1[1],':',color=[0,0.4470,0.7410])
        plt.plot([0,1],[0,1],'--',color=[0.5,0.5,0.5])
        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.xticks([0,0.5,1])
        plt.yticks([0,0.5,1])
        labels = ['Main - '+method2, 'Side - '+method2, 'Main - '+method1, 'Side - '+method1]
        plt.legend(loc='lower right',handles=[newMain[0],newSide[0],oldMain[0],oldSide[0]],fontsize=12,labels=labels)
        if title==1: plt.title(" ".join(self.filename.split("_")[:-1]),fontsize=20)
        
    def rayquot(self, m):
        if not m.isPCA:
            print('Function only available for PCA models')
            return
        return [round((la.norm(self.data[self.sideResp.astype(bool)].dot(m.B))/\
                       la.norm(self.data[self.sideResp.astype(bool)]))*100,2),\
                round((la.norm(self.data[~self.sideResp.astype(bool)].dot(m.B))/\
                       la.norm(self.data[~self.sideResp.astype(bool)]))*100,2)]
    
    def cluster(self, perc, numiter, dat=None, equiSize=True, strict=False):
        if dat is None:
            dat = self.data
            numPnt = self.numPoints
        else:
            numPnt = dat.shape[0]
        std = np.std(dat,axis=0)[None,:]
        std[np.where(std==0)] = 1
        dat = dat/np.ones((numPnt,1)).dot(std)
        k = int(perc*numPnt)
        clusters = dat[np.random.choice(range(numPnt),k,replace=False)]
        for i in range(numiter):
            clusterTemp = np.zeros((k,self.numFields))
            clusterSize = np.zeros(k).astype(int)
            fullClusters = []
            for point in dat:
                best = np.argmin(np.sum(np.delete(clusters-np.ones((k,1)).dot(point[None,:]),fullClusters,axis=0)**2,axis=1))
                if equiSize and not strict: best = np.delete(range(k),fullClusters)[best]
                clusterTemp[best] += point
                clusterSize[best] += 1
                if equiSize and not strict:
                    if len(fullClusters)<numPnt-k*int(1/perc):
                        if clusterSize[best]>int(1/perc): fullClusters.append(best)
                    else:
                        if clusterSize[best]>=int(1/perc): fullClusters.append(best)
            if not equiSize:
                clusterTemp[np.where(clusterSize==0)] = copy.deepcopy(clusters[np.where(clusterSize==0)])
                clusterSize[np.where(clusterSize==0)] = 1
            clusters = copy.deepcopy(clusterTemp/clusterSize[:,None].dot(np.ones((1,self.numFields))))
            if equiSize and strict:
                distMat = np.sqrt(np.sum((dat[:,None,:].repeat(k,axis=1)\
                                          -clusters[None,:,:].repeat(numPnt,axis=0))**2,axis=2))
                distMat = np.hstack((distMat[:,:numPnt-k*int(1/perc)].repeat(int(1/perc)+1,axis=1),\
                                             distMat[:,numPnt-k*int(1/perc):].repeat(int(1/perc),axis=1)))
                clusterToPoints = linear_sum_assignment(distMat.T)[1]
                last = 0
                for j in range(numPnt-k*int(1/perc)):
                    clusters[j] = np.mean(dat[clusterToPoints[last:last+int(1/perc)+1]],axis=0)
                    last += int(1/perc)+1
                for j in range(k-numPnt+k*int(1/perc)):
                    clusters[j] = np.mean(dat[clusterToPoints[last:last+int(1/perc)]],axis=0)
                    last += int(1/perc)
        clusters *= np.ones((k,1)).dot(std)
        return clusters
    
def AUCcomp(prob, numiters = 5, dlist=[0,0.001,0.005,0.01,0.02,0.05,0.1],\
            mulist=[0.001,0.01,0.1,1],lams=np.logspace(-3,3,10)):
    
    zsvmDat = np.zeros((len(dlist),2))
    ssvmDat = np.zeros((len(mulist),len(dlist),2))
    
    for iteration in range(numiters):
        print('starting iteration',iteration,'..')
        prob.splitData(0.7)
        for i,d in zip(range(len(dlist)),dlist):
            print('starting d=',d)
            zsvmDat[i] += prob.ROCStats(prob.zsvm(d,lams=lams,split=True)[0],split=True)
            for j,mu in zip(range(len(mulist)),mulist):
                ssvmDat[j,i] += prob.ROCStats(prob.ssvm(d,mu,lams=lams,split=True)[0],split=True)
        zsvmDat = zsvmDat/numiters
        ssvmDat = ssvmDat/numiters
    
    fig = plt.figure()
    plt.plot(zsvmDat[:,1],zsvmDat[:,0],color='red')
    [plt.plot(ssvmDat[i,:,1],ssvmDat[i,:,0],'--',color='blue') for i in range(len(mulist))]
    plt.title('AUC')
    return fig

def sensitivityPlot(prob, zpcaDat=None, spcaDat=None, k=5, normPCA=True, kernels=['Linear','Gaussian'], numPC=2,\
                    dlist=[0,0.5], mulist=[0.01,0.05,0.1,1], lams=np.logspace(-4,4,5),\
                    figsize=(6,9),title=None):
    
    if zpcaDat is None or spcaDat is None:
        zpcaDat = np.empty((len(dlist),len(kernels)+1))
        spcaDat = np.empty((len(mulist),len(dlist),len(kernels)+1))
        
        for i,d in enumerate(dlist):
            print('starting d=',d)
            err, varExp, totVar, eig, corr = crossVal(prob,k,numPC=numPC,d=d,normPCA=normPCA,kernels=kernels,covCon=False,lams=lams,outputFlag=False)
            zpcaDat[i] = copy.deepcopy(np.hstack((np.array([np.mean(100*varExp/totVar)]),np.mean(err,axis=0).flatten())).flatten())
            for j,mu in zip(range(len(mulist)),mulist):
                print('starting mu=',mu)
                err, varExp, totVar, eig, corr = crossVal(prob,k,numPC=numPC,d=d,mu=mu,normPCA=normPCA,kernels=kernels,lams=lams,outputFlag=False)
                spcaDat[j,i] = copy.deepcopy(np.hstack((np.array([np.mean(100*varExp/totVar)]),np.mean(err,axis=0).flatten())).flatten())
    
    fig = plt.figure(figsize=figsize)
    plt.plot(zpcaDat[:,1],zpcaDat[:,0],color='red')
    [plt.plot(spcaDat[i,:,1],spcaDat[i,:,0],'--',color='blue') for i in range(len(mulist))]
    [plt.text(max(spcaDat[i,:,1])+0.01,max(spcaDat[i,:,0])+0.01,'$\mu=$%s'%mu) for i,mu in enumerate(mulist)]
    plt.xlabel('$\Delta(\mathcal{F})$')
    plt.ylabel('% Variance Explained')
    plt.xlim(plt.axes().axes.get_xlim()[0],plt.axes().axes.get_xlim()[1]+0.1)
    plt.title(title)
    return fig, zpcaDat, spcaDat

def normalize(data):
    l = data.shape[0]
    means = np.ones((l,l)).dot(data)/l
    stdevs = np.sqrt(np.var(data,axis=0))[None,:]
    stdevs[np.where(stdevs==0)] = 1
    return (data-means)*np.ones((l,1)).dot(1/stdevs)+means, means, stdevs

def pcaPlots(prob, kernels=['Linear'], numPC=2, d=0, mu=1, lams=None, linCon=True, covCon=True, predBnd=150, N=25, perc=0.5, split=False):

    print('Fair PCA Parameters: numPC=%s, d=%s, mu=%s'%(numPC,d,mu))
    prob.data, means, stdevs = normalize(prob.data)
    if split: prob.train, means, stdevs = normalize(prob.train)
    if linCon and not covCon: B,m = prob.zpca(dimPCA=numPC,d=d,split=split)
    elif covCon: B,m = prob.spca(dimPCA=numPC,addLinear=linCon,mu=mu,d=d,split=split)
    else: B,m = prob.pca(dimPCA=numPC,split=split)
    eigVecs = la.eigh(m.m.dat.T.dot(m.m.dat))[1][:,-numPC:]
    totVar = np.trace(m.m.dat.T.dot(m.m.dat))
    varExp = np.trace(B.T.dot(m.m.dat.T).dot(m.m.dat.dot(B)))

    print(m.m.m.getprosta(mosek.soltype.itr))
    if m.m.m.getprosta(mosek.soltype.itr)!=mosek.prosta.prim_and_dual_feas: return
    print('Top eigenvalues of solution:',np.round(la.eigvalsh(m.m.X)[-numPC:],4))
    print('Correlation with top eigenvectors:',[round(la.norm(b,2),2) for b in B.T.dot(eigVecs)])
    print('Proportion of variance explained:', varExp/totVar)
    print('Proportion of deviation explained:', np.sqrt(varExp/totVar))
    if linCon or covCon:
        if np.max(np.abs((m.m.B.T.dot(m.getZCon(prob.trainSide if split else prob.sideResp).reshape((prob.numFields,1))))))>1e-7:
            print('Linear constraint unsatisfied')
        else:
            print('Linear constraint satisfied')
    
    prob.data = normalize(prob.data.dot(B))[0]
    if split: prob.test, means, stdevs = normalize(((prob.test-means[:len(prob.test)])/stdevs[:len(prob.test)]+means[:len(prob.test)]).dot(B))
    prob.numFields = 2
    kernelDat = []
    for kernel in kernels:
        print(kernel,'SVM:')
        if kernel=='Linear': w, svm = prob.svm(useSRSP=True,outputFlag=True,split=split,lams=lams)[:2]
        elif kernel=='Gaussian': w, svm = prob.svm(useSRSP=True,dual=True,kernel=lambda x,y: math.exp(-la.norm(x-y)**2/2),outputFlag=True,split=split,lams=lams)[:2]
        elif kernel=='Polynomial': w, svm = prob.svm(useSRSP=True,dual=True,conic=False,kernel=lambda x,y: (x.T.dot(y)+1)**2,outputFlag=True,split=split,lams=lams)[:2]
        else:
            print('\tIncorrect kernel name')
            continue
        
        if svm.dual:
            pred = svm.K.dot(svm.alpha.flatten()*(2*svm.rsp-1))
            err, b = maxROC(svm,prob.test if split else prob.data,prob.testSide if split else prob.sideResp,pred)
            if numPC>1:
                predIdx = np.argsort((pred-b)**2)
                threshPts = prob.data[predIdx[:predBnd]]
                #func = lambda x: svm.alpha.T.dot(svm.getK(x))
                #threshPts = np.array([newton_krylov(func,point) for point in threshPts])
                x = threshPts[:,0]
                y = threshPts[:,1]
                meanx = np.mean(x)
                meany = np.mean(y)
                x -= meanx
                y -= meany
                idx = np.argsort(np.arctan2(x,y))
                x = x[idx] + meanx
                y = y[idx] + meany
                x = np.hstack((x,x[:N]))
                y = np.hstack((y,y[:N]))
                cumsum = np.cumsum(np.insert(x, 0, 0)) 
                x = (cumsum[N:] - cumsum[:-N]) / float(N)
                cumsum = np.cumsum(np.insert(y, 0, 0))
                y = (cumsum[N:] - cumsum[:-N]) / float(N)
        else:
            err, b = maxROC(svm,prob.test if split else prob.data,prob.testSide if split else prob.sideResp)
            if numPC>1:
                dat = prob.test if split else prob.data
                x = np.sort(dat[:,0])[[int(0.02*len(dat)),int(0.98*len(dat))]]
                y = np.sort(dat[:,1])[[int(0.02*len(dat)),int(0.98*len(dat))]]
                x = 0.8*(x-np.mean(x))+np.mean(x)
                y = 0.8*(y-np.mean(y))+np.mean(y)
                if x[1]-x[0]>y[1]-y[0]:
                    x = [(b-svm.B[1]*y[0])/svm.B[0],(b-svm.B[1]*y[1])/svm.B[0]]
                    if max(x)-max(dat[:,0])>0.1*(max(dat[:,0])-min(dat[:,0])):
                        idx = np.argmax(x)
                        x[idx] = 1.1*max(dat[:,0])-0.1*min(dat[:,0])
                        y[idx] = (b-svm.B[0]*x[idx])/svm.B[1]
                    if min(dat[:,0])-min(x)>0.1*(max(dat[:,0])-min(dat[:,0])):
                        idx = np.argmin(x)
                        x[idx] = 1.1*min(dat[:,0])-0.1*max(dat[:,0])
                        y[idx] = (b-svm.B[0]*x[idx])/svm.B[1]
                else:
                    y = [(b-svm.B[0]*x[0])/svm.B[1],(b-svm.B[0]*x[1])/svm.B[1]]
                    if max(y)-max(dat[:,1])>0.1*(max(dat[:,1])-min(dat[:,1])):
                        idx = np.argmax(y)
                        y[idx] = 1.1*max(dat[:,1])-0.1*min(dat[:,1])
                        x[idx] = (b-svm.B[1]*y[idx])/svm.B[0]
                    if min(dat[:,1])-min(y)>0.1*(max(dat[:,1])-min(dat[:,1])):
                        idx = np.argmin(y)
                        y[idx] = 1.1*min(dat[:,1])-0.1*max(dat[:,1])
                        x[idx] = (b-svm.B[1]*y[idx])/svm.B[0]
        #x = (x-means[0,0])*stdevs[0,0]+means[0,0]; y = (y-means[0,1])*stdevs[0,1]+means[0,1];
        main,side,xthresh,ythresh = prob.plot(svm,perc=0.25,thresholds=[b],graph=False,split=split)
        kernelDat.append(kern(kernel,x,y,main,side,xthresh,ythresh))
    graphs(prob.test if split else prob.data,prob.testSide if split else prob.sideResp,kernelDat,perc)
    winsound.PlaySound("SystemExit", winsound.SND_ALIAS)
    return prob.test if split else prob.data, prob.testSide if split else prob.sideResp, B, kernelDat, w

def crossVal(prob, k=1, kernels=['Linear','Gaussian'], numPC=2, d=0, mu=1, normPCA=True, lams=None, linCon=True, covCon=True, dualize=True, outputFlag=True):
    
    idx = np.arange(prob.numPoints); np.random.shuffle(idx)
    
    errList = np.empty((k,len(kernels)))
    varExpList = np.empty(k)
    totVarList = np.empty(k)
    eigList = [[]]*k
    corrList = [[]]*k
    #train_test = np.array([train_test_split((idx),test_size=0.3) for i in range(k)])
    #trainFolds, testFolds = [list(train_test[:,0]),list(train_test[:,1])]
    testFolds = [idx[int(i*prob.numPoints/k):int((i+1)*prob.numPoints/k)] for i in range(k)]
    trainFolds = [np.concatenate([testFolds[j] for j in range(k) if j!=i]) for i in range(k)]
    dat = copy.deepcopy(prob.data)
    srsp = copy.deepcopy(prob.sideResp)
    if outputFlag: print('#################################################################')
    for iteration,(train,test) in enumerate(zip(trainFolds,testFolds)):
        if outputFlag: print('Iteration',iteration)
        if outputFlag: print('Fair PCA Parameters: numPC=%s, d=%s, mu=%s'%(numPC,d,mu))
        if normPCA: prob.data, means, stdevs = normalize(dat[train])
        else: prob.data = dat[train]
        prob.sideResp = srsp[train]
        if linCon and not covCon: B,m = prob.zpca(dimPCA=numPC,d=d)
        elif covCon: B,m = prob.spca(dimPCA=numPC,addLinear=linCon,mu=mu,d=d,dualize=dualize)
        else: B,m = prob.pca(dimPCA=numPC)
        eigVecs = la.eigh(m.m.dat.T.dot(m.m.dat))[1][:,-numPC:]
        varExp = np.trace(B.T.dot(m.m.dat.T).dot(m.m.dat.dot(B)))
        totVar = np.trace(m.m.dat.T.dot(m.m.dat))
        eig = np.round(la.eigvalsh(m.m.X)[-numPC:],4)
        corr = [round(la.norm(b,2),2) for b in B.T.dot(eigVecs)]
        
        varExpList[iteration] = varExp
        totVarList[iteration] = totVar
        eigList[iteration] = eig
        corrList[iteration] = corr
    
        if outputFlag: print(m.m.m.getprosta(mosek.soltype.itr))
        if m.m.m.getprosta(mosek.soltype.itr) not in [mosek.prosta.prim_and_dual_feas,mosek.prosta.unknown]: return None, None, None, None, None
        if outputFlag:
            print('Top eigenvalues of solution:',eig)
            print('Correlation with top eigenvectors:',corr)
            print('Proportion of variance explained:', varExp/totVar)
            print('Proportion of deviation explained:', np.sqrt(varExp/totVar))
            if linCon or covCon:
                if np.max(np.abs((m.m.B.T.dot(m.getZCon(prob.sideResp).reshape((prob.numFields,1))))))>1e-7:
                    print('Linear constraint unsatisfied')
                else:
                    print('Linear constraint satisfied')
        
        if normPCA: prob.data = normalize(((dat[test]-means[:len(test)])/stdevs[:len(test)]+means[:len(test)]).dot(B))[0]
        else: prob.data = normalize(dat[test].dot(B))[0]
        prob.sideResp = srsp[test]
        for kernum,kernel in enumerate(kernels):
            if outputFlag: print(kernel,'SVM:')
            if kernel=='Linear': svm, err = prob.svm(useSRSP=True,outputFlag=outputFlag,lams=lams)[1:]
            elif kernel=='Gaussian': svm, err = prob.svm(useSRSP=True,dual=True,kernel=lambda x,y: math.exp(-la.norm(x-y)**2/2),outputFlag=outputFlag,lams=lams)[1:]
            elif kernel=='Polynomial': svm, err = prob.svm(useSRSP=True,dual=True,conic=False,kernel=lambda x,y: (x.T.dot(y)+1)**2,outputFlag=outputFlag,lams=lams)[1:]
            else:
                if outputFlag: print('\tIncorrect kernel name')
                continue
            errList[iteration,kernum] = err
    if outputFlag:
        print('-----------------------------------------------------------------')
        print('Average variation explained:',np.mean(varExpList/totVarList))
        print('Average deviation explained:',np.mean(np.sqrt(varExpList/totVarList)))
        print('Average errors',np.round(np.mean(errList,axis=0),4))
        winsound.PlaySound("SystemExit", winsound.SND_ALIAS)
    prob.data = dat
    prob.sideResp = srsp
    return errList, varExpList, totVarList, eigList, corrList

def graphs(dat, srsp, kernelDat, perc=0.25, figsize=(6,6), title=True, axes=True):
    numPC = dat.shape[1]
    linType = {'Linear':'-', 'Gaussian':'--', 'Polynomial':'-.-'}
    dat1 = dat[srsp.astype(bool)]
    dat0 = dat[~srsp.astype(bool)]
    #dat1 = (np.eye(dat1.shape[0])-np.ones((dat1.shape[0],dat1.shape[0]))/dat1.shape[0]).dot(dat1)
    #dat0 = (np.eye(dat0.shape[0])-np.ones((dat0.shape[0],dat0.shape[0]))/dat0.shape[0]).dot(dat0)
    rand1 = np.random.choice(range(len(dat1)),int(perc*len(dat1)),replace=False)
    rand0 = np.random.choice(range(len(dat0)),int(perc*len(dat0)),replace=False)
    fig1 = plt.figure(figsize=figsize)
    if numPC>1:
        han1 = plt.plot(dat1[rand1,0],dat1[rand1,1],'.',color='blue',label='protected class 1',ms=5)[0]
        han0 = plt.plot(dat0[rand0,0],dat0[rand0,1],'x',color='orange',label='protected class 0',ms=5)[0]
    hanline = [han1,han0]
    for kernel in kernelDat:
        if numPC>1:
            plt.figure(fig1.number)
            hanline.append(plt.plot(kernel.xbound,kernel.ybound,linType[kernel.name],color='red',lw=2,label=kernel.name+' classifier')[0])
        graphROC(kernel.main,kernel.side,kernel.xthresh,kernel.ythresh,figsize=figsize)
    plt.figure(fig1.number)
    plt.legend(handles=hanline)
    
def graph3D(dat, srsp, figsize=(9,6), title=False, axes=True):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    srsp = srsp.astype(bool)
    ax.view_init(elev=35, azim=240)
    ax.scatter(dat[srsp,0],dat[srsp,1],dat[srsp,2],'.',c='blue',label='protected class 1',s=3)
    ax.scatter(dat[~srsp,0],dat[~srsp,1],dat[~srsp,2],'x',c='orange',label='protected class 0',s=3)
    if axes:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    if title: ax.set_title('Original Data')
    return fig
    
def saveDat(filename,stuff):
    with open(filename+'.pickle','wb') as file:
        cloudpickle.dump(stuff,file)

def loadDat(filename):
    with open(filename+'.pickle','rb') as file:
        return cloudpickle.load(file)

if __name__ == '__main__':
    datasets = ['Adult_Income_Data.csv',        # 0 32561 (NOT FULL RANK) \cite{Lichman:2013}
                'German_Credit_Data.csv',       # 1 1000 (NOT FULL RANK) \cite{Lichman:2013}
                'Recidivism_Data.csv',          # 2 5279 \cite{angwin2016machine}
                'Wine_Quality_Data.csv',        # 3 6497 \cite{cortez2009modeling}
                'Pima_Diabetes_Data.csv',       # 4 768 \cite{smith1988using}
                'Taiwanese_Credit_Data.csv',    # 5 29623 \cite{yeh2009comparisons}
                'SkillCraft_Data.csv',          # 6 3339 \cite{thompson2013video}
                'Energy_Data.csv',              # 7 768 \cite{tsanas2012accurate}
                'Ecoli_Data.csv',               # 8 333 \cite{horton1996probabilistic}
                'Image_Seg_Data.csv',           # 9 660 \cite{Lichman:2013}
                'Letter_Rec_Data.csv',          # 10 20000 \cite{frey1991letter}
                'Magic_Data.csv',               # 11 19020 \cite{bock2004methods}
                'Parkinsons_Data.csv',          # 12 5875 \cite{Lichman:2013}
                'Steel_Data.csv',               # 13 1941 \cite{Lichman:2013}
                'Statlog_Data.csv',             # 14 3071 \cite{Lichman:2013}
                'Biodeg_Data.csv']              # 15 1055 \cite{mansouri2013quantitative}
     
    for i,dataset in enumerate(datasets):
        prob = problem(dataset)
    
        # For large datasets
        if i in [0,5,10,11]: prob.data = prob.data[:int(0.2*prob.numPoints)]; prob.mainResp = prob.mainResp[:int(0.2*prob.numPoints)]; prob.sideResp = prob.sideResp[:int(0.2*prob.numPoints)]; prob.numPoints = int(0.2*prob.numPoints)
        
        #fig, zpcaDat, spcaDat = sensitivityPlot(prob,k=5,kernels=['Linear','Gaussian'])
        
        k=5
        lams = np.logspace(-4,4,5)
        print('#################### ',prob.filename,' ######################')
        #err, varExp, totVar, eig, corr = crossVal(prob, k, normPCA=True, linCon=False, covCon=False, lams=lams)
        #err1, varExp1, totVar1, eig1, corr1 = crossVal(prob, k, normPCA=True, d=0, covCon=False, lams=lams)
        err2, varExp2, totVar2, eig2, corr2 = crossVal(prob, k, mu=0.1, dualize=False, normPCA=True, d=0, lams=lams, outputFlag=False)
        #stuff = [err,err1,err2,varExp,varExp1,varExp2,totVar,totVar1,totVar2,eig,eig1,eig2,corr,corr1,corr2]
        #print('#############################################################')
        #print('perc variance: %s\nperc deviation: %s\nerrors: %s'%\
        #      (round(100*np.mean(varExp/totVar),2),round(100*np.mean(np.sqrt(varExp/totVar)),2),np.mean(err,axis=0)))
        #print('perc variance: %s\nperc deviation: %s\nerrors: %s'%\
        #      (round(100*np.mean(varExp1/totVar1),2),round(100*np.mean(np.sqrt(varExp1/totVar1)),2),np.mean(err1,axis=0)))
        print('perc variance: %s\nperc deviation: %s\nerrors: %s'%\
              (round(100*np.mean(varExp2/totVar2),2),round(100*np.mean(np.sqrt(varExp2/totVar2)),2),np.round(np.mean(err2,axis=0),4)))
    
    '''
    split = False
    if split: prob.splitData(0.8)
    #idx = np.arange(prob.numPoints); np.random.shuffle(idx); idx = idx[:int(0.2*prob.numPoints)]
    #prob.data = prob.data[idx]; prob.mainResp = prob.mainResp[idx]; prob.sideResp = prob.sideResp[idx]; prob.numPoints = len(idx)
    mean1 = np.array([1,1,2])
    mean2 = np.array([-1,-1,-1])
    cov1 = np.array([[1.0,0.8,0.0],\
                     [0.8,1.0,-1.0],\
                     [0.0,-1.0,3.0]])
    cov2 = np.array([[0.5,0.5,0.0],\
                     [0.5,1.0,-1.2],\
                     [0.0,-1.2,3.0]])
    prob.setGaussData(mean1,mean2,cov1,cov2)
    graph3D(prob.data,prob.sideResp).show()
    prob1 = copy.deepcopy(prob)
    prob2 = copy.deepcopy(prob)
    lams = np.logspace(-4,4,5)
    dat, srsp, B, ker, w = pcaPlots(prob,['Linear','Gaussian'],lams=lams,numPC=2,predBnd=150,perc=1,linCon=False,covCon=False,split=split)
    dat1, srsp1, B1, ker1, w1 = pcaPlots(prob1,['Linear','Gaussian'],lams=lams,numPC=2,d=0,predBnd=150,perc=1,linCon=True,covCon=False,split=split)
    dat2, srsp2, B2, ker2, w2 = pcaPlots(prob2,['Linear','Gaussian'],lams=lams,numPC=2,d=0,mu=1,predBnd=150,perc=1,linCon=True,covCon=True,split=split)
    stuff = [dat,dat1,dat2,srsp,srsp1,srsp2,B,B1,B2,ker,ker1,ker2,mean1,mean2,cov1,cov2]
    #dat,dat1,dat2,srsp,srsp1,srsp2,B,B1,B2,ker,ker1,ker2,mean1,mean2,cov1,cov2 = loadDat('randData')
    '''