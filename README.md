# CogSci-2017

Source code for analyses reported in

Musslick, S., Saxe, M. A., Özcimder K., Dey, B., Henselman, G., Cohen, J. D. (submitted). Multitasking Capability Versus Learning Efficiency in Neural Network Architectures. Proceedings of the 39th Annual Meeting of the Cognitive Science Society.

## Simulation 1

The simulation involves training a neural networks on a set of tasks with
different initial weight conditions. Each initial weight condition
determines the initial similarity between tasks that rely on the same
input dimension. At each training iteration the learning performance, as
well as multitasking performance are assessed.

To generate a replication of the stimulation index by 'rep', run
Simulation1(rep)

E.g. Simulation1(1)

Results averaged across multiple replications can be plotted using the code in
Simulation1_plots.m
