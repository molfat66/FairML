# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 01:23:19 2017

@author: mahbo

Original SDP formulation of the fair SVM problem (NOT USED)

"""
import copy, mosek, numpy as np

def dualSDP(self,d,presolveTol=1.0e-30,split=False,outputFlag=False):
    dat, mrsp, srsp, numPnt = self.checkIfSplit()
    
    # Make mosek environment
    with mosek.Env() as env:
    
        # Create a task object and attach log stream printer
        with env.Task(0,0) as task:
            task.putdouparam(mosek.dparam.presolve_tol_x,presolveTol)
            if outputFlag: task.set_Stream(mosek.streamtype.msg, streamprinter)
    
            c = [srsp[i]/sum(srsp)+(1-srsp[i])/sum(1-srsp) for i in range(numPnt)]
            M = self.M
    
            # Bound keys for constraints
            bkc = [mosek.boundkey.lo]*(3*numPnt) + [mosek.boundkey.ra, mosek.boundkey.fx, mosek.boundkey.fx]
    
            # Bound values for constraints
            blc = [1.0]*numPnt + c + [0.0]*numPnt + [-d, 0.0, 1.0]
            buc = [+math.inf]*(3*numPnt) + [d, 0.0, 1.0]
    
            # Below is the sparse representation of the A
            # matrix stored by row. 
            asub = [[2*numPnt+i] for i in range(numPnt)] + [[numPnt+i] for i in range(numPnt)]\
            + [[i] for i in range(numPnt)] + [list(range(2*numPnt)), [], []]
            aval = [[1.0]]*(3*numPnt) + [[M]*numPnt + [1.0]*numPnt, [], []]
    
            barci   = list(range(self.numFields))
            barcj   = list(range(self.numFields))
            barcval = [1.0]*self.numFields
    
            barai   = [[self.numFields+numPnt]*self.numFields]*numPnt + [[self.numFields+numPnt]]*(2*numPnt)\
            + [[self.numFields+i for i in range(numPnt) for j in range(self.numFields)], [self.numFields+numPnt]*numPnt,
               [self.numFields+numPnt]]
            baraj   = [list(range(self.numFields))]*numPnt + [[self.numFields+i] for i in range(numPnt)]*2\
            + [[j for i in range(numPnt) for j in range(self.numFields)], [self.numFields+i for i in range(numPnt)],
               [self.numFields+numPnt]]
            baraval = [list(0.5*mrsp[i]*dat[i]) for i in range(numPnt)] + [[0.5*M]]*numPnt + [[-0.5]]*numPnt\
            + [[0.5*dat[i,j] for i in range(numPnt) for j in range(self.numFields)], [0.5]*numPnt, [1.0]]
    
            numvar = 3*numPnt
            numcon = len(bkc)
            BARVARDIM = [self.numFields+numPnt+1]
    
            # Append 'numvar' variables.
            # The variables will initially be fixed at zero (x=0). 
            task.appendvars(numvar)
    
            # Append 'numcon' empty constraints.
            # The constraints will initially have no bounds. 
            task.appendcons(numcon)
    
            # Append matrix variables of sizes in 'BARVARDIM'.
            # The variables will initially be fixed at zero. 
            task.appendbarvars(BARVARDIM)
    
            # Set the linear term c_0 in the objective.
            task.putclist([2*numPnt+i for i in range(numPnt)], [1.0]*numPnt)
            symc = task.appendsparsesymmat(BARVARDIM[0], barci, barcj, barcval)
            task.putbarcj(0, [symc], [1.0])
    
            for j in range(numvar):
                # Set the bounds on variable j
                # blx[j] <= x_j <= bux[j] 
                task.putvarbound(j, mosek.boundkey.lo, 0, +math.inf)
    
            syma = []
            for i in range(numcon):
                # Set the bounds on constraints.
                task.putconbound(i, bkc[i], blc[i], buc[i])
                # Input row i of A 
                task.putarow(i, asub[i], aval[i])
                # add coefficient matrix of PSD variables
                syma.append(task.appendsparsesymmat(BARVARDIM[0], barai[i], baraj[i], baraval[i]))
                task.putbaraij(i, 0, [syma[i]], [1.0])
    
            # Input the objective sense (minimize/maximize)
            task.putobjsense(mosek.objsense.minimize)
    
            # Solve the problem and print summary
            task.optimize()
            task.solutionsummary(mosek.streamtype.msg)
    
            # Get status information about the solution
            prosta = task.getprosta(mosek.soltype.itr)
            solsta = task.getsolsta(mosek.soltype.itr)
    
            if (solsta == mosek.solsta.optimal or 
              solsta == mosek.solsta.near_optimal):
                xx = [0.]*numvar
                task.getxx(mosek.soltype.itr, xx)
                lenbarvar = BARVARDIM[0] * (BARVARDIM[0]+1) / 2
                barx = [0.]*int(lenbarvar)
                task.getbarxj(mosek.soltype.itr, 0, barx)
                barxx = [barx[(self.numFields+numPnt+1)*(i+1)-1-sum(list(range(i+1)))]\
                              for i in range(self.numFields+numPnt)]
                
                alpha = xx[:numPnt]
                beta = np.array(barxx[:self.numFields]).reshape((self.numFields,1))
                gamma = barxx[-numPnt:]
                lam = xx[numPnt:2*numPnt]
                eps = xx[-numPnt:]
                Gamma21 = np.array([[barx[self.numFields+i+(self.numFields+numPnt+1)*j-sum(list(range(j+1)))]\
                                          for j in range(self.numFields)] for i in range(numPnt)])
                time = task.getdouinf(mosek.dinfitem.optimizer_time)
                print("Optimal solution = %s\nruntime = %s seconds" % (beta.T.dot(beta)[0,0]+sum(eps),time))
                return beta, alpha, gamma, lam, eps, Gamma21
            elif (solsta == mosek.solsta.dual_infeas_cer or
                solsta == mosek.solsta.prim_infeas_cer or
                solsta == mosek.solsta.near_dual_infeas_cer or
                solsta == mosek.solsta.near_prim_infeas_cer):
                print("Primal or dual infeasibility certificate found.\n")
                return None, None, None, None, None, None
            elif solsta == mosek.solsta.unknown:
                print("Unknown solution status")
                return None, None, None, None, None, None
            else:
                print("Other solution status")
                return None, None, None, None, None, None