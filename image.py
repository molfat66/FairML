# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 15:24:47 2017

@author: mahbo

Generates a dummy plot to motivate covariance constraints (NOT USED)

"""

import math, matplotlib.pyplot as plt, numpy as np

phi1 = 1.25*math.pi
phi2 = 1.95*math.pi
xlen1 = 2.5
ylen1 = 1
xlen2 = 3
ylen2 = 1
shift1 = [2,1]
shift2 = [-3,0.5]
colMult = 0.5

thetas = np.linspace(0,2*math.pi,360)
x = np.sin(thetas).reshape((len(thetas),1))
y = np.cos(thetas).reshape((len(thetas),1))
xy = np.hstack((x,y))
rotMat1 = np.array([[xlen1*np.cos(phi1),-xlen1*np.sin(phi1)],[ylen1*np.sin(phi1),ylen1*np.cos(phi1)]])
rotMat2 = np.array([[xlen2*np.cos(phi2),-xlen2*np.sin(phi2)],[ylen2*np.sin(phi2),ylen2*np.cos(phi2)]])
lin = lambda x: (shift2[1]-shift1[1])/(shift2[0]-shift1[0])*x
pnts1 = np.random.multivariate_normal(shift1,0.1*rotMat1.T.dot(rotMat1),150)
pnts2 = np.random.multivariate_normal(shift2,0.1*rotMat2.T.dot(rotMat2),150)

plt.plot(xy.dot(rotMat1)[:,0]+shift1[0],xy.dot(rotMat1)[:,1]+shift1[1],color=[0,0.4470+colMult*(1-0.4470),0.7410+colMult*(1-0.7410)])
plt.plot(xy.dot(rotMat2)[:,0]+shift2[0],xy.dot(rotMat2)[:,1]+shift2[1],color=[0.8500+colMult*(1-0.8500),0.3250+colMult*(1-0.3250),0.0980+colMult*(1-0.0980)])
plt.fill(xy.dot(rotMat1)[:,0]+shift1[0],xy.dot(rotMat1)[:,1]+shift1[1],color=[0,0.4470+colMult*(1-0.4470),0.7410+colMult*(1-0.7410)])
plt.fill(xy.dot(rotMat2)[:,0]+shift2[0],xy.dot(rotMat2)[:,1]+shift2[1],color=[0.8500+colMult*(1-0.8500),0.3250+colMult*(1-0.3250),0.0980+colMult*(1-0.0980)])
plt.plot(pnts1[:,0],pnts1[:,1],'.')
plt.plot(pnts2[:,0],pnts2[:,1],'.')
plt.plot(shift1[0],shift1[1],'.',color='black',ms=10)
plt.plot(shift2[0],shift2[1],'.',color='black',ms=10)
plt.text(shift1[0],shift1[1]+0.1,'$\mathbf{X_+}$',size=20)
plt.text(shift2[0],shift2[1]+0.15,'$\mathbf{X_-}$',size=20)
plt.plot(range(-6,5),[lin(x) for x in range(-6,5)],'--',color='r',lw=3)