# SRO
This repo contains implementations and solvers of the Stochastic Resource Optimization (SRO) problem. 

A stochastic resource is a generator that creates a value with a given random variable $X$.
Additionally, values are associated with a cost function $c$.
The optimization problem is:

Given a set of resources $RES$, a probability target $p_{target}$ and a value target $v_{target}$:

$$
\begin{alignat}{2}
        & \underset{x \in 2^{R}}{\operatorname{arg\,min}} & \mathbb{E}(c(x))                                \\
        & \text{subject to } & 1 - cdf(\sum x)(v_{target}) \geq p_{target}
\end{alignat}
$$

Here $\sum x$ refers to the joint distribution of the given set of resources.

# State of Development
Currently this repo is in a very early stage of development.
It contains two main variants:
* resources modeled by continuous random variables with correlations given by copulas
* independent resources modeled by discrete random variables

Solvers are distinguished between optimal solvers, heuristic solvers, and distributed solvers.

# Related Packages
* [Mango.jl](https://github.com/OFFIS-DAI/Mango.jl) - for modeling agents for distributed solvers
* [Copulas.jl](https://github.com/lrnv/Copulas.jl) - for implementing correlations