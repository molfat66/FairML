This package contains MATLAB implementations of the spectral algorithms for
computing fair support vector machines, which were developed in the paper

M. Olfat and A. Aswani (2017). Spectral algorithms for computing fair
support vector machines. 

Some additional details are listed below:

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
