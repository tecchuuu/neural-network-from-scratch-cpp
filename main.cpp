#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

struct Neuron {
    double bias;
    std::vector<double> weights;
};

enum activation {RELU, SIGMOID};
std::vector<double> layerNeuron(const std::vector<Neuron>& n, const std::vector<double>& x, activation);
double rectifiedLinearUnit(double x1);
double sigmoid(double x1);


int main() {
    double x1 = -6;
    double x2 = 10;
    double x3 = -10;
    std::vector<double> input = {x1, x2, x3};

    std::vector<Neuron> neurons = {
        {1, {0.5, 0.5, 0.5}},
        {0.5, {0.8, 0.4, 0.2}},
        {0.8, {0.3, 0.6, 0.7}}
    };

    std::vector<double> output = layerNeuron(neurons, input, SIGMOID);

    for (const double& v: output) {
        std::cout << v << std::endl;
    }

    return 0;
}

std::vector<double> layerNeuron(const std::vector<Neuron>& n, const std::vector<double>& x, activation act) {
    std::vector<double> output;
    size_t numInputs = x.size();
    size_t numNeurons = n.size();
    
    for (size_t i = 0; i < numNeurons; i++) {
        double weighted_x = n[i].bias;
        for (size_t j = 0; j < numInputs; j++) {
            weighted_x += x.at(j) * n[i].weights[j];
        }
        switch (act) {
            case RELU:
                weighted_x = rectifiedLinearUnit(weighted_x);
                break;
            case SIGMOID:
                weighted_x = sigmoid(weighted_x);
                break;
        }

        output.push_back(weighted_x);
    }
    
    return output;
}

double rectifiedLinearUnit(double x1) {
    if (x1 < 0) {
        return 0;
    } 
    
    return x1;
}

double sigmoid(double x1) {
    return 1 / (1 + exp(-x1));
}