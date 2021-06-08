# Neural ODE useful stuff

## Main Idea

The main idea is to represent a neural network as numerical method for solving a differential equation (ex: [Euler method](https://en.wikipedia.org/wiki/Euler_method)).

Advantages (which shortly presented [here](https://towardsdatascience.com/neural-odes-breakdown-of-another-deep-learning-breakthrough-3e78c7213795)]:

* Memory efficiency: don’t need to store the parameters while backpropagating
* Adaptive computation: balance speed and accuracy with the discretization scheme, moreover, having it different while training and inference
* Parameters efficiency: the parameters of nearby “layers” are automatically tied together
* Normalizing flows new type of invertible density models
* Continuous time series models: continuously-defined dynamics can naturally incorporate data which arrives at arbitrary times


## Links

### Repos: 

* [**awesome**](https://github.com/Zymrael/awesome-neural-ode)
* [Neural Ordinary Differential Equations](https://github.com/rtqichen/torchdiffeq) // *authors' original, Differentiable ODE solvers with full GPU support and O(1)-memory backpropagation, Pytorch*
* [Simple Pytorch implementation of Neural Ordinary Differential Equations](https://github.com/msurtsukov/neural-ode) // *(2019, with explanations)*

### Videos:

* [Neural ODE, Dmitry Vetrov](https://www.youtube.com/watch?v=8yJekeeGp_I) (rus)
* [Neural Differential Equationsm, Siraj Raval](https://www.youtube.com/watch?v=AD3K8j12EIE&t=1487s)

### Papers:

TODO

### Other

* [towardsdatascience](https://towardsdatascience.com/neural-odes-breakdown-of-another-deep-learning-breakthrough-3e78c7213795)