# neural-network-from-scratch-cpp

A neural network built from the ground up in C++ — no ML/math libraries (no Eigen, no PyTorch, nothing that does the matrix math or the network logic for you). The goal is understanding, not shortcuts: every neuron, every layer, and eventually every gradient is hand-written.

"From scratch" here means the neural network logic and linear algebra are entirely custom. Standard C++ tools (`iostream`, `std::vector`, `cmath`, `random`) are used as plumbing, the same way you'd use `+` or `printf` — they don't do any of the ML thinking.

## Current status

- Single neuron: weighted sum of inputs + bias
- Activation functions: ReLU and sigmoid, selectable via an `enum` and applied per layer
- A full layer: arbitrary number of neurons, arbitrary number of inputs, weights/bias bundled per-neuron via a `Neuron` struct
- Not yet implemented: multi-layer chaining, random weight initialization, loss function, backpropagation, gradient descent, training loop

## Building

Requires a C++ compiler (developed with MinGW-w64 g++).

```
g++ main.cpp -o main
./main
```

## Roadmap

1. Single neuron
2. Layer of neurons
3. Chain multiple layers into a full forward pass
4. Random weight initialization
5. Loss function
6. Backpropagation (chain rule / partial derivatives)
7. Gradient descent + training loop
8. Train on a real dataset
