#include <iostream>
#include <vector>
#include "mnist_loader.h"
#include "cnn.h"
#include "conv_layer.h"
#include "pooling_layer.h"
#include "fully_connected_layer.h"


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
    
    // Cargar los datos de entrenamiento y prueba
    std::vector<std::vector<float>> train_images, test_images;
    std::vector<int> train_labels, test_labels;
    
    // Rutas a los archivos de datos MNIST
    std::string train_images_path = "../data/train-images.idx3-ubyte";
    std::string train_labels_path = "../data/train-labels.idx1-ubyte";
    std::string test_images_path = "../data/t10k-images.idx3-ubyte";  // Archivo de prueba
    std::string test_labels_path = "../data/t10k-labels.idx1-ubyte";  // Etiquetas de prueba

    // Cargar imágenes y etiquetas de entrenamiento
    loadMNISTImages(train_images_path, train_images);
    loadMNISTLabels(train_labels_path, train_labels);

    // Cargar imágenes y etiquetas de prueba
    loadMNISTImages(test_images_path, test_images);
    loadMNISTLabels(test_labels_path, test_labels);

    float learning_rate = 0.01;
    int epochs = 20;

    // *** Entrenamiento ***
    for (int epoch = 0; epoch < epochs; ++epoch) {
        float total_loss = 0.0;
        int correct_predictions = 0;

        for (size_t i = 0; i < train_images.size(); ++i) {
            // Convertir la imagen de 1D a 2D (28x28)
            std::vector<std::vector<float>> mnist_image = reshape_image(train_images[i], 28, 28);

            // Forward pass
            auto output = cnn.forward(mnist_image);

            // Crear la etiqueta como one-hot encoding
            std::vector<int> target(10, 0);
            target[train_labels[i]] = 1;

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

            // Verificar si la predicción es correcta
            int predicted_label = std::distance(output.begin(), std::max_element(output.begin(), output.end()));
            if (predicted_label == train_labels[i]) {
                correct_predictions++;
            }
        }

        // Calcular precisión en los datos de entrenamiento
        float accuracy = (float)correct_predictions / train_images.size();
        std::cout << "Epoch " << epoch + 1 << " completed. Loss: " << total_loss / train_images.size()
                  << " | Accuracy: " << accuracy * 100.0f << "%" << std::endl;
    }

    // *** Evaluación con los datos de prueba ***
    std::cout << "Evaluating on test data..." << std::endl;
    float test_accuracy = evaluate(cnn, test_images, test_labels);
    std::cout << "Test Accuracy: " << test_accuracy * 100.0f << "%" << std::endl;

    return 0;
}
