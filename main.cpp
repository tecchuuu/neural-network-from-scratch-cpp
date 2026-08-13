#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <random>

struct Neuron {
    double bias;
    std::vector<double> weights;
};

struct LayerResult {
    std::vector<double> preActivationWeightedX;
    std::vector<double> postActivationWeightedX;
};

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<double> dist(-1.0, 1.0);
enum activation {RELU, SIGMOID};
LayerResult layerNeuron(const std::vector<Neuron>& n, const std::vector<double>& x, activation);
double rectifiedLinearUnit(double x1);
double sigmoid(double x1);
std::vector<Neuron> makeLayer(int numNeurons, int numInputPerNeuron);
double meanSquaredError(const std::vector<double>& output, const std::vector<double>& target);
double weightGradient(double output, double target, double weighted_x, double x, activation act);
double biasGradient(double output, double target, double weighted_x, activation act);

int main() {
    std::vector<double> input = {-6, 10, -10};
    std::vector<Neuron> hiddenLayer = makeLayer(5, 3);
    std::vector<double> target = {2, 3};

    for (size_t i = 0; i < hiddenLayer.size(); i++) {
        std::cout << "{" << hiddenLayer[i].bias << " ,{";
        for (size_t j = 0; j < hiddenLayer[i].weights.size(); j++) {
            std::cout << hiddenLayer[i].weights[j];
            if (j + 1 < hiddenLayer[i].weights.size()) std::cout << ", ";
        }
        std::cout << "}}" << std::endl;
    }
    std::cout << "\n" << std::endl;

    LayerResult hiddenLayerOutput = layerNeuron(hiddenLayer, input, SIGMOID);

    for (size_t i = 0; i < hiddenLayerOutput.preActivationWeightedX.size(); i++) {
        std::cout << "pre: " << hiddenLayerOutput.preActivationWeightedX[i] << ", post: "
            << hiddenLayerOutput.postActivationWeightedX[i] << std::endl;
    }
    std::cout << "\n" << std::endl;

    std::vector<Neuron> hiddenLayer2 = makeLayer(2, 5);

    LayerResult hiddenLayer2Output = layerNeuron(hiddenLayer2, hiddenLayerOutput.postActivationWeightedX, RELU);

    for (size_t i = 0; i < hiddenLayer2Output.preActivationWeightedX.size(); i++) {
        std::cout << "pre2: " << hiddenLayer2Output.preActivationWeightedX[i] << ", post2: "
            << hiddenLayer2Output.postActivationWeightedX[i] << std::endl;
    }

    std::cout << "\nmean squared error : " << meanSquaredError(hiddenLayer2Output.postActivationWeightedX, target) << "\n" << std::endl;

    for (int i = 0; i < 3; i++) {
        size_t neuronSize = hiddenLayer2.size();
        size_t weightSize = hiddenLayer2[0].weights.size();
        double learningRate = 0.01;

        for (size_t j = 0; j < neuronSize; j++) {
            hiddenLayer2[j].bias -= learningRate * biasGradient(hiddenLayer2Output.postActivationWeightedX[j], target[j], hiddenLayer2Output.preActivationWeightedX[j], SIGMOID);
            for (size_t k = 0; k < weightSize; k++) {
                hiddenLayer2[j].weights[k] -= learningRate * weightGradient(hiddenLayer2Output.postActivationWeightedX[j], target[j], hiddenLayer2Output.preActivationWeightedX[j], hiddenLayerOutput.postActivationWeightedX[k], SIGMOID);
                std::cout << "hiddenLayerWeights " << hiddenLayer2[j].weights[k] << std::endl;
            }
            std::cout << "hiddenLayerBias " << hiddenLayer2[j].bias  << "\n" << std::endl;
        }
        hiddenLayer2Output = layerNeuron(hiddenLayer2, hiddenLayerOutput.postActivationWeightedX, SIGMOID);
        std::cout << "mean squared error : " << meanSquaredError(hiddenLayer2Output.postActivationWeightedX, target) << "\n" << std::endl;
    }

    return 0;
}

LayerResult layerNeuron(const std::vector<Neuron>& n, const std::vector<double>& x, activation act) {
    LayerResult layerResult;
    size_t numInputs = x.size();
    size_t numNeurons = n.size();
    for (size_t i = 0; i < numNeurons; i++) {
        assert(x.size() == n[i].weights.size());
        double preActivationWeightedX;
        double postActivationWeightedX;
        preActivationWeightedX = n[i].bias;
        for (size_t j = 0; j < numInputs; j++) {
            preActivationWeightedX += x.at(j) * n[i].weights[j];   
        }
        layerResult.preActivationWeightedX.push_back(preActivationWeightedX);
        switch (act) {
            case RELU:
                postActivationWeightedX = rectifiedLinearUnit(preActivationWeightedX);
                break;
            case SIGMOID:
                postActivationWeightedX = sigmoid(preActivationWeightedX);
                break;
        }
        layerResult.postActivationWeightedX.push_back(postActivationWeightedX);
    }

    return layerResult;
}

double rectifiedLinearUnit(double x1) {
    if (x1 < 0) {return 0;} return x1;
}

double sigmoid(double x1) {
    return 1 / (1 + exp(-x1));
}

std::vector<Neuron> makeLayer(int numNeurons, int numInputPerNeuron) {
    std::vector<Neuron> dynamicNeurons;

    for (size_t i = 0; i < numNeurons; i++) {
        Neuron neuronBox;
        double randomBias = dist(gen);
        neuronBox.bias = randomBias;
        for (size_t j = 0; j < numInputPerNeuron; j++) {
            double randomWeights = dist(gen);
            neuronBox.weights.push_back(randomWeights);
        }
        dynamicNeurons.push_back(neuronBox);
    }    

    return dynamicNeurons;
}

double meanSquaredError(const std::vector<double>& output, const std::vector<double>& target) {
    assert(output.size() == target.size());
    double MSE = 0;
    for (size_t i = 0; i < output.size(); i++) {
        MSE += std::pow(output[i] - target[i], 2);
    }

    return MSE / output.size();
}

double weightGradient(double output, double target, double weighted_x, double x, activation act) {
    switch (act) {
        case SIGMOID:
            return 2 * (output - target) * (output * (1 - output)) * x;
        case RELU:
            if (weighted_x <= 0) {
                return  0;
            } else { 
                return 2 * x * (output - target);
            }
        default: return 0;
    }
}

double biasGradient(double output, double target, double weighted_x, activation act) {
    return weightGradient(output, target, weighted_x, 1, act);
}