# FairML

The following contains the code for the fair SVM and fair PCA procedures, as well as some the datasets used (after some cleaning). All data downloaded from the UC Irvine online Machine Learning Repository (citation below). Descriptions of the individual code files are as follows:

problem.py: A problem object is the main unit of action in this package. It encapsulates the data for any problem, and is the conduit through which various fair and unfair SVM and PCA algorithms can be run on the data, as well as the conduit through which the results of these algorithms may be plotted and analyzed. Also contains procedures for testing algorithm and generating plots of interest used in the paper.

model.py: A model object is the conduit through which all optimization is done. It stand to handle the different types of optimization objects from a unified framework from the perspective of the problem object. It has the added benefit of being a self contained optimization, allowing for cross-validation or comparisons at the level of the problem object without having to define all of the extra variables and constraint coefficients associated with each individual optimization

mosPCAMult.py: Mosek implementation of FPCA that handles all desired PC's at once. Records all constraints and symmetric matrices defined in the task object for aiding in debugging.

mosSVM.py: Mosek implementation of fair SVM, with functionality for both kernal and linear SVM. Functionality also provided for iterative fair SVM procedure, although this is not used.

dataImport.py: Imports and cleans NHANES data, runs unconstrained PCA, FPCA with only the mean constraint, and FPCA with both constraints on it, projects onto the resulting subspaces, conducts clustering on the separate sets of projected data, returns the compostion of the clusters in terms of the protected attribute (>40 years old vs <=40) and plots the means of each of the clusters.

mosPCA.py: Mosek implementation of FPCA that can only find one PC. Functionality is provided to quickly project data in anticipation of use in finding multiple PC's.

image.py: Generates a dummy plot to motivate covariance constraints (NOT USED).

gurMod.py: Gurobi implementation of older versions of fair SVM algorithm (NOT USED).

SVM_Extras.py: Original SDP formulation of the fair SVM problem (NOT USED).

Lichman, M. (2013). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science. 
