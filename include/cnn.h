#ifndef CNN_H
#define CNN_H

#include "conv_layer.h"
#include "pooling_layer.h"
#include "fully_connected_layer.h"
#include <vector>

template<typename T>
class CNN {
private:
    ConvLayer<T> conv1;
    PoolingLayer<T> pool1;
    FullyConnectedLayer<T> fc1;
    int input_size;
    int conv_output_size;
    int pool_output_size;

public:
        CNN(int input_size = 28) : 
        input_size(input_size),
        conv1(8, 3), 
        pool1(2),
        conv_output_size(input_size - conv1.getFilterSize() + 1),
        pool_output_size(0),  // Inicializamos a 0 temporalmente
        fc1(0, 10)  // Inicializamos con 0 temporalmente
{ 
    pool_output_size = pool1.getOutputSize(conv_output_size);
    fc1 = FullyConnectedLayer<T>(pool_output_size * pool_output_size * 8, 10);
}
    std::vector<T> forward(const std::vector<std::vector<T>>& input) {
        auto conv_output = conv1.forward(input);
        auto pool_output = pool1.forward(conv_output);

        // Aplanar la salida de pooling para pasarla a la capa fully connected
        std::vector<T> flattened;
        for (const auto& row : pool_output) {
            flattened.insert(flattened.end(), row.begin(), row.end());
        }

        return fc1.forward(flattened);
    }

    // Método de retropropagación
    void backward(const std::vector<T>& d_output, const std::vector<std::vector<T>>& input, T learning_rate) {
        std::vector<T> flattened_pool_output;
        auto pool_output = pool1.forward(conv1.forward(input));
        for (const auto& row : pool_output) {
           flattened_pool_output.insert(flattened_pool_output.end(), row.begin(), row.end());
        }
        
        
        // Retropropagación en la capa fully connected
            std::vector<T> d_pool_output = fc1.backward(d_output, flattened_pool_output, learning_rate);


        // Reshape de d_pool_output a la forma adecuada para la capa de pooling
        std::vector<std::vector<T>> d_conv_output(pool_output_size, std::vector<T>(pool_output_size));
        for (int i = 0; i < pool_output_size; ++i) {
            for (int j = 0; j < pool_output_size; ++j) {
                d_conv_output[i][j] = d_pool_output[i * pool_output_size + j];
            }
        }

        // Retropropagación en la capa de pooling
        std::vector<std::vector<T>> d_input = pool1.backward(d_conv_output);

        // Retropropagación en la capa convolucional
        conv1.backward(d_input, input, learning_rate);
    }
};

// Función para calcular la pérdida cross-entropy
template<typename T>
T cross_entropy_loss(const std::vector<T>& predicted, const std::vector<int>& labels) {
    T loss = 0.0;
    for (size_t i = 0; i < predicted.size(); ++i) {
        loss -= (labels[i] == 1) ? std::log(predicted[i] + 1e-9) : 0;
    }
    return loss;
}

#endif // CNN_H