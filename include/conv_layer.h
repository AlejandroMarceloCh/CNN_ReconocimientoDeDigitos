#ifndef CONV_LAYER_H
#define CONV_LAYER_H

#include <vector>
#include <random>

// Usamos un template para soportar diferentes tipos de datos
template<typename T>
class ConvLayer {
private:
    int num_filters;
    int filter_size;
    std::vector<std::vector<std::vector<T>>> filters;

    // Función para inicializar los filtros con valores aleatorios
    void initialize_filters() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<T> dis(-0.1, 0.1);

        for (int f = 0; f < num_filters; ++f) {
            for (int i = 0; i < filter_size; ++i) {
                for (int j = 0; j < filter_size; ++j) {
                    filters[f][i][j] = dis(gen);
                }
            }
        }
    }

public:
    ConvLayer(int num_filters, int filter_size) : num_filters(num_filters), filter_size(filter_size) {
        filters.resize(num_filters, std::vector<std::vector<T>>(filter_size, std::vector<T>(filter_size)));
        initialize_filters();
    }

    // Operación de convolución (forward pass)
    std::vector<std::vector<T>> forward(const std::vector<std::vector<T>>& input) {
        int input_size = input.size();
        int output_size = input_size - filter_size + 1;
        std::vector<std::vector<T>> output(output_size, std::vector<T>(output_size, 0.0));

        for (int f = 0; f < num_filters; ++f) {
            for (int i = 0; i < output_size; ++i) {
                for (int j = 0; j < output_size; ++j) {
                    T sum = 0.0;
                    for (int fi = 0; fi < filter_size; ++fi) {
                        for (int fj = 0; fj < filter_size; ++fj) {
                            sum += input[i + fi][j + fj] * filters[f][fi][fj];
                        }
                    }
                    output[i][j] = sum;
                }
            }
        }
        return output;
    }

// Retropropagación para la capa convolucional
    std::vector<std::vector<T>> backward(const std::vector<std::vector<T>>& d_output, const std::vector<std::vector<T>>& input, T learning_rate) {
        // Inicializar gradientes para los filtros
        std::vector<std::vector<std::vector<T>>> d_filters(num_filters, std::vector<std::vector<T>>(filter_size, std::vector<T>(filter_size, 0.0)));
        std::vector<std::vector<T>> d_input(input.size(), std::vector<T>(input[0].size(), 0.0));

        for (int f = 0; f < num_filters; ++f) {
            for (int i = 0; i < d_output.size(); ++i) {
                for (int j = 0; j < d_output[0].size(); ++j) {
                    for (int fi = 0; fi < filter_size; ++fi) {
                        for (int fj = 0; fj < filter_size; ++fj) {
                            // Actualizar los filtros con gradiente descendente
                            d_filters[f][fi][fj] += d_output[i][j] * input[i + fi][j + fj];
                            d_input[i + fi][j + fj] += filters[f][fi][fj] * d_output[i][j];
                        }
                    }
                }
            }
        }

        // Actualizar los filtros con el gradiente
        for (int f = 0; f < num_filters; ++f) {
            for (int fi = 0; fi < filter_size; ++fi) {
                for (int fj = 0; fj < filter_size; ++fj) {
                    filters[f][fi][fj] -= learning_rate * d_filters[f][fi][fj];
                }
            }
        }

        return d_input;
    }
    // En conv_layer.h, dentro de la clase ConvLayer
        int getFilterSize() const { return filter_size; }



};

#endif // CONV_LAYER_H
