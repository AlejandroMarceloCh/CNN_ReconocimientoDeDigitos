#ifndef FULLY_CONNECTED_LAYER_H
#define FULLY_CONNECTED_LAYER_H

#include <vector>
#include <cmath>
#include <random>

// Usamos templates para soportar diferentes tipos de datos (float, double, etc.)
template<typename T>
class FullyConnectedLayer {
private:
    std::vector<std::vector<T>> weights;
    std::vector<T> biases;

    void initialize_weights() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<T> dis(-0.1, 0.1);

        for (int i = 0; i < weights.size(); ++i) {
            for (int j = 0; j < weights[i].size(); ++j) {
                weights[i][j] = dis(gen);
            }
            biases[i] = dis(gen);
        }
    }

public:
    FullyConnectedLayer(int input_size, int output_size) {
        weights.resize(output_size, std::vector<T>(input_size));
        biases.resize(output_size);
        initialize_weights();
    }

    // Forward pass
    std::vector<T> forward(const std::vector<T>& input) {
        std::vector<T> output(biases.size(), 0.0);
        for (int i = 0; i < output.size(); ++i) {
            for (int j = 0; j < input.size(); ++j) {
                output[i] += input[j] * weights[i][j];
            }
            output[i] += biases[i];
            output[i] = sigmoid(output[i]);  // Aplicar la función de activación
        }
        return output;
    }

    T sigmoid(T x) {
        return 1.0 / (1.0 + std::exp(-x));
    }
};

#endif // FULLY_CONNECTED_LAYER_H
