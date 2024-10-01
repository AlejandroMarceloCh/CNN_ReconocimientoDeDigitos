#ifndef POOLING_LAYER_H
#define POOLING_LAYER_H

#include <vector>
#include <algorithm>
#include <cmath>
#include <limits> // Para std::numeric_limits

// Usamos templates para permitir diferentes tipos de datos
template<typename T>
class PoolingLayer {
private:
    int pool_size;
    std::vector<std::vector<T>> last_input; // Almacenar la última entrada

public:
    PoolingLayer(int pool_size) : pool_size(pool_size) {}

    // Max pooling
    std::vector<std::vector<T>> forward(const std::vector<std::vector<T>>& input) {
        last_input = input; // Guardar la entrada original
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

    // Método de retropropagación
    std::vector<std::vector<T>> backward(const std::vector<std::vector<T>>& d_output) {
        int input_size = d_output.size() * pool_size; // Tamaño de la entrada original
        std::vector<std::vector<T>> d_input(input_size, std::vector<T>(input_size, 0.0));

        for (int i = 0; i < d_output.size(); ++i) {
            for (int j = 0; j < d_output[0].size(); ++j) {
                // Encontrar el índice del valor máximo en la ventana de pooling
                int max_i = i * pool_size;
                int max_j = j * pool_size;
                for (int pi = 0; pi < pool_size; ++pi) {
                    for (int pj = 0; pj < pool_size; ++pj) {
                        // Solo el valor máximo recibe el gradiente
                        if (last_input[max_i + pi][max_j + pj] == d_output[i][j]) { // Cambiado input por last_input
                            d_input[max_i + pi][max_j + pj] += d_output[i][j]; // Distribuir el gradiente
                        }
                    }
                }
            }
        }
        return d_input;
    }

    // Método para obtener el tamaño de salida
    int getOutputSize(int input_size) {
        return input_size / pool_size;
    }
};

#endif // POOLING_LAYER_H
