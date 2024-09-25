#ifndef POOLING_LAYER_H
#define POOLING_LAYER_H

#include <vector>
#include <algorithm>
#include <cmath>

// Usamos templates para permitir diferentes tipos de datos
template<typename T>
class PoolingLayer {
private:
    int pool_size;

public:
    PoolingLayer(int pool_size) : pool_size(pool_size) {}

    // Max pooling
    std::vector<std::vector<T>> forward(const std::vector<std::vector<T>>& input) {
        int input_size = input.size();
        int output_size = input_size / pool_size;
        std::vector<std::vector<T>> output(output_size, std::vector<T>(output_size, 0.0));

        for (int i = 0; i < output_size; ++i) {
            for (int j = 0; j < output_size; ++j) {
                T max_value = -std::numeric_limits<T>::infinity();
                for (int pi = 0; pi < pool_size; ++pi) {
                    for (int pj = 0; pj < pool_size; ++pj) {
                        max_value = std::max(max_value, input[i * pool_size + pi][j * pool_size + pj]);
                    }
                }
                output[i][j] = max_value;
            }
        }
        return output;
    }
};

#endif // POOLING_LAYER_H
