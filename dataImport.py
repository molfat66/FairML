# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 18:57:00 2018

@author: mahbo

Imports and cleans NHANES data, runs unconstrained PCA, FPCA with only the mean
constraint, and FPCA with both constraints on it, projects onto the resulting subspaces,
conducts clustering on the separate sets of projected data, returns the compostion of the
clusters in terms of the protected attribute (>40 years old vs <=40) and plots the
means of each of the clusters

"""

import xport, os.path, numpy as np, pandas as pd, matplotlib.pyplot as plt
from problem import *
from scipy.stats import percentileofscore
from sklearn.cluster import KMeans

dd = None
df = None
prob = None

def loadAndCleanData(overWrite=False):
    global dd, df
    with open('C:\\Users\\mahbo\\Documents\\Health_Data_2005-2006\\DEMOGRAPHIC_D.xpt', 'rb') as f:
        dd = pd.DataFrame([row[:9] for row in xport.Reader(f)],\
                           columns=['SEQN','SDDSRVYR','RIDSTATR','RIDEXMON','RIAGENDR',\
                                    'RIDAGEYR','RIDAGEMN','RIDAGEEX','RIDRETH1'])
    dd.drop(['SDDSRVYR','RIDSTATR','RIDEXMON','RIDAGEMN','RIDAGEEX'],axis=1,inplace=True)
    dd.set_index('SEQN',inplace=True)
    #with open('C:\\Users\\mahbo\\Documents\\Health_Data_2005-2006\\MEDICAL_D.xpt', 'rb') as f:
    #    datamed = [row for row in xport.Reader(f)]
    
    if not os.path.isfile('datapax.pickle') or overWrite:
        with open('C:\\Users\\mahbo\\Documents\\Health_Data_2005-2006\\paxraw_d.xpt', 'rb') as f:
            df = pd.DataFrame([row for row in xport.Reader(f) if row[3] not in [1.0,7.0]],\
                               columns=['SEQN','PAXSTAT','PAXCAL','PAXDAY','PAXN','PAXHOUR',\
                                        'PAXMINUT','PAXINTEN','PAXSTEP'])
    
        df = df[(df.PAXSTAT==1.0)&(df.PAXCAL==1.0)] # removes <5mm rows
        df.PAXN = ((df.PAXN-1)%1400)+1
        df['PAXBUCK'] = 10*np.floor((df.PAXN-1)/10)
        df.drop(['PAXSTAT','PAXCAL','PAXN','PAXSTEP'],axis=1,inplace=True)
        df = df.pivot_table(index='SEQN',columns='PAXBUCK',values='PAXINTEN')
        saveDat('datapax',df)
    else:
        df = loadDat('datapax')
    cols = df.columns
    for i in range(int(len(cols)/2)):
        df[cols[int(2*i+1)]] += df[cols[int(2*i)]]
        df.drop(cols[int(2*i)],axis=1,inplace=True)
    
    #plt.hist(df.sum(axis=1))
    df = df[df.sum(axis=1)<=np.percentile(df.sum(axis=1),99)] # removes all with sum >~18k
    #plt.hist((df==0.0).sum(axis=1))
    df = df[(df==0.0).sum(axis=1)<100] # removes <5% of data

def runPCAs(mu=0.1, numPC=2, overWrite=False):
    global prob, df, dd
    
    prob = problem()
    prob.setData(np.array(df),np.array(dd.RIDAGEYR.loc[list(df.index)]>40).astype(int))
    
    if not os.path.isfile('basis.pickle') or overWrite:
        B,m = prob.pca(dimPCA=numPC)
        B1,m1 = prob.zpca(dimPCA=numPC)
        B2,m2 = prob.spca(dimPCA=numPC,mu=mu,dualize=True)
        saveDat('basis',[B,B1,B2])
    else:
        B,B1,B2 = loadDat('basis')
    return B, B1, B2

def getClustering(B, B1, B2, k=4, overWrite=False):
    if not os.path.isfile('clusters.pickle') or overWrite:
        clust = KMeans(n_clusters=k).fit_predict(prob.data.dot(B))
        clust1 = KMeans(n_clusters=k).fit_predict(prob.data.dot(B1))
        clust2 = KMeans(n_clusters=k).fit_predict(prob.data.dot(B2))
        saveDat('clusters',[clust,clust1,clust2])
    else:
        clust,clust1,clust2 = loadDat('clusters')
    return clust, clust1, clust2

def getClusterProportions(clust,clust1,clust2):
    global df
    
    df['SRSP'] = prob.sideResp
    df['CL'] = clust
    df['CL1'] = clust1
    df['CL2'] = clust2
    
    print(df.SRSP.mean())
    print(df.groupby('CL').SRSP.mean())
    print(df.groupby('CL1').SRSP.mean())
    print(df.groupby('CL2').SRSP.mean())
    df = df/1000

def singlePlot(figsize=(5,5), fontsize=9, ylim=1.55, save=False):
    means = np.array(df.drop(['SRSP','CL1','CL2'],axis=1).groupby('CL').mean())
    means1 = np.array(df.drop(['SRSP','CL','CL2'],axis=1).groupby('CL1').mean())
    means2 = np.array(df.drop(['SRSP','CL1','CL'],axis=1).groupby('CL2').mean())
    
    devs = np.array(df.drop(['SRSP','CL1','CL2'],axis=1).groupby('CL').std())/np.sqrt(np.array(df.drop(['SRSP','CL1','CL2'],axis=1).groupby('CL').count()))
    devs1 = np.array(df.drop(['SRSP','CL','CL2'],axis=1).groupby('CL1').std())/np.sqrt(np.array(df.drop(['SRSP','CL','CL2'],axis=1).groupby('CL1').count()))
    devs2 = np.array(df.drop(['SRSP','CL1','CL'],axis=1).groupby('CL2').std())/np.sqrt(np.array(df.drop(['SRSP','CL1','CL'],axis=1).groupby('CL2').count()))
    
    fig = plt.figure(figsize=figsize)
    axcommon = fig.add_subplot(111)
    axcommon.spines['top'].set_color('none')
    axcommon.spines['bottom'].set_color('none')
    axcommon.spines['left'].set_color('none')
    axcommon.spines['right'].set_color('none')
    axcommon.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
    axcommon.set_xlabel('Minutes',fontsize=fontsize)
    axcommon.set_ylabel("Intensity (thousands)",fontsize=fontsize)
    [item.set_fontsize(fontsize) for item in axcommon.get_xticklabels()+axcommon.get_yticklabels()]
    #axcommon.yaxis.set_label_coords(-0.1,0.5)

    ax = fig.add_subplot(311)
    plt.setp(ax.get_xticklabels(), visible=False)
    ax.set_ylim(0,ylim)
    [plt.errorbar(df.columns[:-4],mean,yerr=dev,lw=1) for mean,dev in zip(means[[1,0,2]],devs)]
    
    ax1 = fig.add_subplot(312,sharex=ax)
    plt.setp(ax1.get_xticklabels(), visible=False)
    ax1.set_ylim(0,ylim)
    [plt.errorbar(df.columns[:-4],mean,yerr=dev,lw=1) for mean,dev in zip(means1,devs1)]
    
    ax2 = fig.add_subplot(313,sharex=ax)
    [plt.errorbar(df.columns[:-4],mean,yerr=dev,lw=1) for mean,dev in zip(means2[[2,0,1]],devs2)]
    ax2.set_ylim(0,ylim)
    if save: fig.savefig('PCA_Latex/nhanesClustersAll.png',dpi=3*fig.dpi)
    
    return fig

def separatePlots(figsize=(2.5,4), fontsize=9, ylim=1.55, save=False):
    means = np.array(df.drop(['SRSP','CL1','CL2'],axis=1).groupby('CL').mean())
    means1 = np.array(df.drop(['SRSP','CL','CL2'],axis=1).groupby('CL1').mean())
    means2 = np.array(df.drop(['SRSP','CL1','CL'],axis=1).groupby('CL2').mean())
    
    devs = np.array(df.drop(['SRSP','CL1','CL2'],axis=1).groupby('CL').std())/np.sqrt(np.array(df.drop(['SRSP','CL1','CL2'],axis=1).groupby('CL').count()))
    devs1 = np.array(df.drop(['SRSP','CL','CL2'],axis=1).groupby('CL1').std())/np.sqrt(np.array(df.drop(['SRSP','CL','CL2'],axis=1).groupby('CL1').count()))
    devs2 = np.array(df.drop(['SRSP','CL1','CL'],axis=1).groupby('CL2').std())/np.sqrt(np.array(df.drop(['SRSP','CL1','CL'],axis=1).groupby('CL2').count()))
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.set_xlabel('Minutes',fontsize=9)
    ax.set_ylabel("Intensity (thousands)",fontsize=fontsize)
    ax.set_ylim(0,ylim)
    [item.set_fontsize(fontsize) for item in ax.get_xticklabels()+ax.get_yticklabels()]
    lines = [plt.errorbar(df.columns[:-4],mean,yerr=dev,lw=1,color=[0,0.4470,0.7410]) for mean,dev in zip(means,devs)]
    if save: fig.savefig('PCA_Latex/nhanesClustersNone.png',dpi=2*fig.dpi)
    
    fig1 = plt.figure(figsize=figsize)
    ax = fig1.add_subplot(111)
    ax.set_xlabel('Minutes',fontsize=9)
    ax.set_ylabel("Intensity (thousands)",fontsize=fontsize)
    ax.set_ylim(0,ylim)
    [item.set_fontsize(fontsize) for item in ax.get_xticklabels()+ax.get_yticklabels()]
    lines = [plt.errorbar(df.columns[:-4],mean,yerr=dev,lw=1,color=[0,0.4470,0.7410]) for mean,dev in zip(means1,devs1)]
    if save: fig1.savefig('PCA_Latex/nhanesClustersOne.png',dpi=2*fig.dpi)
    
    fig2 = plt.figure(figsize=figsize)
    ax = fig2.add_subplot(111)
    ax.set_xlabel('Minutes',fontsize=9)
    ax.set_ylabel("Intensity (thousands)",fontsize=fontsize)
    ax.set_ylim(0,ylim)
    [item.set_fontsize(fontsize) for item in ax.get_xticklabels()+ax.get_yticklabels()]
    lines = [plt.errorbar(df.columns[:-4],mean,yerr=dev,lw=1,color=[0,0.4470,0.7410]) for mean,dev in zip(means2,devs2)]
    if save: fig2.savefig('PCA_Latex/nhanesClustersBoth.png',dpi=2*fig.dpi)
    
    return fig, fig1, fig2

if __name__ == '__main__':
    loadAndCleanData()
    B,B1,B2 = runPCAs(numPC=5)
    clust,clust1,clust2 = getClustering(B,B1,B2,k=3)
    getClusterProportions(clust,clust1,clust2)
    fig = singlePlot(ylim=1.65)
    