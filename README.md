# Deep learning based Surrogate for optimization
These codes are used to construct the deep learning based surrogate models for fast and accurate optimization. 

The codes in 1.1 - 1.3 optimize discrete variables, locations of wells for fluid production, using CNN-based surrogate models.

The codes in 2.2 - 2.3 optimize time dependent continuous variables, operation conditions, using LSTM-based surrogate models. 

The details of CNN-based surrogate model can be found in my co-authored paper: Kim, J., Yang, H., & Choe, J. (2020). Robust optimization of the locations and types of multiple wells using CNN based proxy models. Journal of Petroleum Science and Engineering, 107424.

https://www.sciencedirect.com/science/article/pii/S0920410520304964

# Methodology for CNN based proxy 

1. Time of flight (TOF) map: The TOF which means the travel time of fluids along streamlines provides quantities about convective flow in the reservoir. The velocity field is derived from the solution to the pressure distribution and application of Darcy’s law. Because streamlines consider reservoir structure, heterogeneity, and geometry of well patterns in its calculation, each information is implicit in the TOF. 

2. Convolutional neural network (CNN) 

<img src="https://user-images.githubusercontent.com/72419213/97830633-76d61580-1c82-11eb-94e3-ca1c314e0857.gif" width="500">

3. Particle Swarm optimization (PSO) : PSO is a population-based stochastic global search method. It was first introduced by Kennedy and Eberhardt (1995) and applied for the optimization of well type and location by Onwunalu and Durlofsky (2010). PSO algorithm begins with N_p number of potential solutions called particles, which randomly initialized in hyper-dimensional search space. A set of particles, referred to as a swarm, explore through the search space by changing the position of each particle according to its own memory and that of its neighbors. 

4. Retraining of surrogate model during optimization 

When we construct a surrogate model, we use a few sample data simulated by a full-physics reservoir simulator. As the optimization process goes on, the search space for each generation is likely to move away from the initial space where sample data are generated for the proxy modeling. This makes the proxy model underestimate the objective function values, which impedes the optimization algorithm to distinguish the best position among previously explored positions. To prevent deviations from the proxy predictions and the true results, the proxy model should be retrained using new sample data that are obtained in the middle of the optimization process. 

<img src="https://user-images.githubusercontent.com/72419213/97830963-8bff7400-1c83-11eb-9325-a21467824ddd.jpg" width="500">

# Results for CNN based proxy 

1. 2D synthetic model 

For the optimization using PSO, it is important to find the best solution with the highest objective function value in each generation and compare it with the previous best solution. Below figure shows taht the optimization with proxy model presents a similar average result with that with the reservoir simulation model. 

<img src="https://user-images.githubusercontent.com/72419213/97831185-265fb780-1c84-11eb-8264-eef31013a67c.png" width="300">

The total computational cost for the optimization using the surrogate model is 6,612 seconds, which is approximately 19.3% of the total time for the optimization using the reservoir simulation model. 

<img src="https://user-images.githubusercontent.com/72419213/97831227-3f686880-1c84-11eb-8d44-734f2835cd29.PNG" width="500">

2. 3D egg model 

Below Figure shows the average of NPV expectations from both forward models with the number of generations. Given the random effects of the PSO, the proxy model provides the results similar to those using the reservoir simulation model. 

<img src="https://user-images.githubusercontent.com/72419213/97831610-6a06f100-1c85-11eb-8572-68893704b115.png" width="300">

The optimized scenario is shown as 

<img src="https://user-images.githubusercontent.com/72419213/97831627-7c812a80-1c85-11eb-8e8d-dfa316a413e0.png" width="300">

In this 3D egg model, because of the efficiency of streamline-based simulation, the average time for obtaining TOF maps is less than 0.15, while running an ECLIPSE requires more than 8 seconds, and thus the computational cost for optimization with reservoir simulation is prohibitively expensive. The remarkable computational efficiency of the proposed surrogate model enables the robust well placement optimization in 3D channelized model with reasonable computation time.

<img src="https://user-images.githubusercontent.com/72419213/97831665-93c01800-1c85-11eb-81c2-8f81c06be223.PNG" width="500">

# Conclusions

- Incorporating time of flight into input data increases the accuracy of proxy model

- Convolutional neural network based proxy additionally improves the performance

- The proxy predicts NPVs with sufficient accuracy, replacing a reservoir simulator

- The proxy enables robust well placement optimization while saving computation time

# Execution details 

See "Execution manual.pdf"
