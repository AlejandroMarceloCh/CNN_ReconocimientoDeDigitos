#ifndef CNN_H
#define CNN_H

#include "conv_layer.h"
#include "pooling_layer.h"
#include "fully_connected_layer.h"

template<typename T>
class CNN {
private:
    ConvLayer<T> conv1;
    PoolingLayer<T> pool1;
    FullyConnectedLayer<T> fc1;

public:
    CNN() : conv1(8, 3), pool1(2), fc1(128, 10) {}

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
};


// Función para calcular la pérdida cross-entropy
template<typename T>
T cross_entropy_loss(const std::vector<T>& predicted, const std::vector<int>& labels) {
    T loss = 0.0;
    for (size_t i = 0; i < predicted.size(); ++i) {
        loss -= labels[i] * std::log(predicted[i] + 1e-9);  // Añadimos un pequeño valor para evitar log(0)
    }
    return loss;
}



#endif // CNN_H
