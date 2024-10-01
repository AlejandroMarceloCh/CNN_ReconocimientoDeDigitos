#include <iostream>
#include <vector>
#include "mnist_loader.h"
#include "cnn.h"
#include "conv_layer.h"

// Función para transformar una imagen de 1D (784 elementos) a 2D (28x28)
std::vector<std::vector<float>> reshape_image(const std::vector<float>& image_1d, int width, int height) {
    std::vector<std::vector<float>> image_2d(height, std::vector<float>(width));
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            image_2d[i][j] = image_1d[i * width + j];
        }
    }
    return image_2d;
}


// Función para evaluar el modelo
float evaluate(CNN<float>& cnn, const std::vector<std::vector<float>>& test_images, const std::vector<int>& test_labels) {
    int correct_predictions = 0;
    
    for (size_t i = 0; i < test_images.size(); ++i) {
        // Convertir imagen de 1D a 2D
        std::vector<std::vector<float>> mnist_image = reshape_image(test_images[i], 28, 28);

        // Forward pass
        auto output = cnn.forward(mnist_image);

        // Obtener la clase predicha
        int predicted_label = std::distance(output.begin(), std::max_element(output.begin(), output.end()));

        // Comparar con la etiqueta real
        if (predicted_label == test_labels[i]) {
            correct_predictions++;
        }
    }

    // Calcular precisión
    float accuracy = (float)correct_predictions / test_images.size();
    return accuracy;
}


int main() {
    CNN<float> cnn;
    
    std::vector<std::vector<float>> images;
    std::vector<int> labels;
    std::string train_images_path = "../data/train-images.idx3-ubyte";
    std::string train_labels_path = "../data/train-labels.idx1-ubyte";

    loadMNISTImages(train_images_path, images);
    loadMNISTLabels(train_labels_path, labels);

    float learning_rate = 0.01;
    int epochs = 20;

    // Ciclo de entrenamiento
    for (int epoch = 0; epoch < epochs; ++epoch) {
        float total_loss = 0.0;
        for (size_t i = 0; i < images.size(); ++i) {
            // Convertir la imagen de 1D a 2D (28x28)
            std::vector<std::vector<float>> mnist_image = reshape_image(images[i], 28, 28);

            // Forward pass
            auto output = cnn.forward(mnist_image);

            // Crear la etiqueta como one-hot encoding
            std::vector<int> target(10, 0);
            target[labels[i]] = 1;

            // Calcular la pérdida
            float loss = cross_entropy_loss(output, target);
            total_loss += loss;

            // Calcular el gradiente de la pérdida respecto a la salida
            std::vector<float> d_output(output.size());
            for (size_t j = 0; j < output.size(); ++j) {
                d_output[j] = output[j] - target[j];
            }

            // Backward pass
            cnn.backward(d_output, mnist_image, learning_rate);
        }

        std::cout << "Epoch " << epoch + 1 << " completed. Loss: " << total_loss / images.size() << std::endl;
    }

    return 0;
}
