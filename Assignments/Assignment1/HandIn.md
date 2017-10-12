#Assignment 1 Hand-In
**Group: cookie**

The goal of this assignment was to build a simple CNN based on the given specifications
and to plot the loss curves of different optimizers.

Our solustion resulted in the following plots:

![Adadelta](/plots/Adadelta.png)

![SGD](/plots/SGD.png)

![Adam](/plots/Adam.png)

We can observe that Stochastic Gradient Descent and Adadelta behave similarly while Adam converges much quicker.

Although we also had some instances where Adam did not converge:
![Adam_non](/plots/Adam_non-convergence.png)