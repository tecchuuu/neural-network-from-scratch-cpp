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
LayerResult computeLayer(const std::vector<Neuron>& neurons, const std::vector<double>& input, activation);
double rectifiedLinearUnit(double x1);
double sigmoid(double x1);
std::vector<Neuron> makeLayer(int numNeurons, int numInputPerNeuron);
double meanSquaredError(const std::vector<double>& output, const std::vector<double>& target);
double weightGradient(double delta, double x);
double outputDeltaSigmoid(double output, double target);
double outputDeltaRelu(double output, double target, double weighted_x);
std::vector<double> hiddenDelta(const std::vector<double>& nextLayerDelta, const std::vector<Neuron>& nextLayerNeurons, const LayerResult& previousLayerResult, activation act);

int main() {
    std::vector<std::vector<double>> inputs = {{0,0},{0,1},{1,0},{1,1}};
    std::vector<std::vector<double>> targets = {{0},{1},{1},{0}};

    std::vector<Neuron> hiddenLayer = makeLayer(4, 2);
    std::vector<Neuron> hiddenLayer2 = makeLayer(1, 4);

    double learningRate = 0.5;

    for (int epoch = 0; epoch < 3001; epoch++) {
        double epochLoss = 0;

        for (size_t s = 0; s < inputs.size(); s++) {
            const std::vector<double>& input = inputs[s];
            const std::vector<double>& target = targets[s];

            LayerResult hiddenLayerOutput = computeLayer(hiddenLayer, input, SIGMOID);
            LayerResult hiddenLayer2Output = computeLayer(hiddenLayer2, hiddenLayerOutput.postActivationWeightedX, SIGMOID);

            epochLoss += meanSquaredError(hiddenLayer2Output.postActivationWeightedX, target);

            // backpropagation
            std::vector<double> deltaHiddenLayer2;
            for (size_t i = 0; i < hiddenLayer2.size(); i++) {
                deltaHiddenLayer2.push_back(outputDeltaSigmoid(hiddenLayer2Output.postActivationWeightedX[i], target[i]));
            }

            std::vector<double> deltaHiddenLayer = hiddenDelta(deltaHiddenLayer2, hiddenLayer2, hiddenLayerOutput, SIGMOID);

            // update weights hiddenLayer
            for (size_t i = 0; i < hiddenLayer.size(); i++) {
                for (size_t j = 0; j < hiddenLayer[i].weights.size(); j++) {
                    hiddenLayer[i].weights[j] -= learningRate * (deltaHiddenLayer[i] * input[j]);
                }
                hiddenLayer[i].bias -= learningRate * deltaHiddenLayer[i];
            }

            // update weights hiddenLayer2
            for (size_t i = 0; i < hiddenLayer2.size(); i++) {
                for (size_t j = 0; j < hiddenLayer2[i].weights.size(); j++) {
                    hiddenLayer2[i].weights[j] -= learningRate * (deltaHiddenLayer2[i] * hiddenLayerOutput.postActivationWeightedX[j]);
                }
                hiddenLayer2[i].bias -= learningRate * deltaHiddenLayer2[i];
            }
        }

        if (epoch % 500 == 0) {
            std::cout << "epoch " << epoch << " : " << epochLoss / inputs.size() << std::endl;
        }
    }

    std::cout << "\nFinal predictions:" << std::endl;
    for (size_t s = 0; s < inputs.size(); s++) {
        LayerResult hiddenLayerOutput = computeLayer(hiddenLayer, inputs[s], SIGMOID);
        LayerResult hiddenLayer2Output = computeLayer(hiddenLayer2, hiddenLayerOutput.postActivationWeightedX, SIGMOID);
        std::cout << inputs[s][0] << " XOR " << inputs[s][1] << " = "
            << hiddenLayer2Output.postActivationWeightedX[0] << " (target " << targets[s][0] << ")" << std::endl;
    }

    return 0;
}

LayerResult computeLayer(const std::vector<Neuron>& neurons, const std::vector<double>& input, activation act) {
    LayerResult layerResult;
    size_t numInputs = input.size();
    size_t numNeurons = neurons.size();
    for (size_t i = 0; i < numNeurons; i++) {
        assert(input.size() == neurons[i].weights.size());
        double preActivationWeightedX;
        double postActivationWeightedX;
        preActivationWeightedX = neurons[i].bias;
        for (size_t j = 0; j < numInputs; j++) {
            preActivationWeightedX += input.at(j) * neurons[i].weights[j];   
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
    return (x1 < 0) ? 0: x1;
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

double weightGradient(double delta, double x) {
    return delta * x;
}

double outputDeltaSigmoid(double output, double target) {
    return (2 * (output - target)) * (output * (1 - output));
}

double outputDeltaRelu(double output, double target, double weighted_x) {
    if (weighted_x <= 0) {
        return  0;
    } else { 
        return (2 * (output - target));
    }
}

std::vector<double> hiddenDelta(const std::vector<double>& nextLayerDelta, const std::vector<Neuron>& nextLayerNeurons, const LayerResult& previousLayerResult, activation act) {
    size_t numOfNextLayerNeurons = nextLayerNeurons.size();
    size_t numOfThisLayerNeurons = previousLayerResult.preActivationWeightedX.size();
    std::vector<double> hiddenLayerDelta;

    for (size_t i = 0; i < numOfThisLayerNeurons; i++) {
        double delta = 0;
        for (size_t j = 0; j < numOfNextLayerNeurons; j++) {
            delta += nextLayerDelta[j] * nextLayerNeurons[j].weights[i];
        }
        switch (act) {
            case SIGMOID:
                delta *= previousLayerResult.postActivationWeightedX[i] * (1 - previousLayerResult.postActivationWeightedX[i]);
                break;
            case RELU:
                if (previousLayerResult.preActivationWeightedX[i] < 0) {
                    delta *= 0;
                }
                break;
        }
        hiddenLayerDelta.push_back(delta);
    }
    return hiddenLayerDelta;
}