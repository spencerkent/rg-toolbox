rg-toolbox
======
A simple implementation of the Radial Gaussianization model of Lyu and
Simoncelli<sup>[[1]](#ref1)</sup>. This paper argues that radial gaussianization
is an effective way to produce a factorial code for multivariate data when that
data comes from an elliptically symmetric distribution. Furthermore it notes that 
because certain representations of images contain multivariate statistics that
are approximately elliptical, this type of transform may be effective at coding
image data. A connection to Divisive Normalization<sup>[[2]](#ref2)</sup><sup>[[3]](#ref3)</sup> 
is made, which is a model of nonlinear coding in populations of neurons.

This implementation is largely untested and a work in progress, use at your own risk :)

### Authors
* Spencer Kent

### References

[<a name="ref1">1</a>]: Lyu, S., & Simoncelli, E. P. (2009).
Nonlinear extraction of independent components of natural images using radial gaussianization. 
_Neural Computation_, 21(6), 1485–1519.

[<a name="ref2">2</a>]: Schwartz, O., & Simoncelli, E. P. (2001).
Natural sound statistics and divisive normalization in the auditory system.
_Advances in Neural Information Processing Systems_.         

[<a name="ref3">3</a>]: Wainwright, M. J., Schwartz, O., & Simoncelli, E. P. (2002). 
Natural Image Statistics and Divisive Normalization. In R.P.N. Rao, B.A. Olshausen, & M.S. Lewicki (Ed.), 
_Probabilistic models of the brain: Perception and neural function_. Cambridge, MA: MIT Press.
