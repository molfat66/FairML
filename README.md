# FairML

This package contains python code for the procedures for calculatinng fair PCA described in the paper

M. Olfat and A. Aswani (2018). Convex Formulations for Fair Principal Component Analysis.

It also contains both MATLAB and python implementations of the spectral algorithms for
computing fair support vector machines, which were developed in the paper

M. Olfat and A. Aswani (2017). Spectral algorithms for computing fair support vector machines.

The authors of may be contacted at:

molfat [at] berkeley [dot] edu
aaswani [at] berkeley [dot] edu

All datasets included were downloaded from the UC Irvine online Machine Learning Repository (citation below). Descriptions of the .py files are as follows:

1. A problem object is the main unit of action for the python implementation in this package. It encapsulates the data for any problem, and is the conduit through which various fair and unfair SVM and PCA algorithms can be run on the data, as well as the conduit through which the results of these algorithms may be plotted and analyzed. problem.py also contains procedures for testing algorithm and generating plots of interest used in the paper.

2. A model object is the conduit through which all optimization is done in the python implementation of this package. It stand to handle the different types of optimization objects from a unified framework from the perspective of the problem object. It has the added benefit of being a self contained optimization, allowing for cross-validation or comparisons at the level of the problem object without having to define all of the extra variables and constraint coefficients associated with each individual optimization

3. The file mosPCAMult.py includes a Mosek implementation of FPCA that handles all desired PC's at once. The object also records all constraints and symmetric matrices defined in the task object for aiding in debugging.

4. The file mosSVM.py inlcludes a Mosek implementation of fair SVM, with functionality for both kernel and linear SVM. Functionality also provided in this file for iterative fair SVM procedure, although this is not referenced in the papers.

5. The file gurMod.py includes a Gurobi implementation of the fair SVM algorithm, although it lacks functionality for handling covariance constraints.

Some additional details regarding the MATLAB code are listed below:

1. The CVX library (available at http://cvxr.com/cvx/) is required.

2. The MOSEK solver (available at https://mosek.com/) is required.

3. The file "main.m" compares three variants of linear SVM

4. The file "kmain.m" compares three variants of kernel SVM

5. The parameters d and mu control the fairness level, and correspond to
   the same variables given in the formulations of the above paper

6. The code uses a wine quality dataset from:

P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. Modeling wine 
preferences by data mining from physicochemical properties. In Decision 
Support Systems, Elsevier, 47(4):547-553, 2009. 

7. The figures show from left to right the ROC plots for the label (blue) 
   and the protected class (red) for regular SVM, linearly constrained 
   SVM, and a spectral SVM 

8. The file "sample_main_figure.pdf" shows a representative example of 
   output from "main.m"

9. The file "sample_kmain_figure.pdf" shows a representative example of 
   output from "kmain.m"

Lichman, M. (2013). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science. 
