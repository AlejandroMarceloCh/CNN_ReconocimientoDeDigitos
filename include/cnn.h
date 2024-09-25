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

#endif // CNN_H
