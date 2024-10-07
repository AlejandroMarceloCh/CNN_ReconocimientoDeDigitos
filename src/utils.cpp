#include "utils.h"
#include <algorithm>
#include <iostream>
#include <fstream>
#include <cmath>

// Reshape un vector de imagen 1D en una matriz de imagen 2D
std::vector<std::vector<float>> reshape_image(const std::vector<float>& image_1d, int width, int height) {
    std::vector<std::vector<float>> image_2d(height, std::vector<float>(width));
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            image_2d[i][j] = image_1d[i * width + j];
        }
    }
    return image_2d;
}

// Evaluar la precisión de un modelo de CNN en un conjunto de imágenes de prueba y etiquetas
float evaluate(CNN<float>& cnn, const std::vector<std::vector<float>>& test_images, const std::vector<int>& test_labels) {
    int correct_predictions = 0;
    for (size_t i = 0; i < test_images.size(); ++i) {
        std::vector<std::vector<float>> mnist_image = reshape_image(test_images[i], 28, 28);
        auto output = cnn.forward(mnist_image);
        int predicted_label = std::distance(output.begin(), std::max_element(output.begin(), output.end()));
        if (predicted_label == test_labels[i]) {
            correct_predictions++;
        }
    }
    return static_cast<float>(correct_predictions) / test_images.size();
}

// Calcular la pérdida de entropía cruzada entre las probabilidades predichas y las etiquetas verdaderas
float cross_entropy_loss(const std::vector<float>& predicted, const std::vector<int>& labels) {
    float loss = 0.0f;
    for (size_t i = 0; i < predicted.size(); ++i) {
        loss -= labels[i] * std::log(predicted[i] + 1e-7f);
    }
    return loss;
}

// Mostrar una barra de progreso que indica el progreso de una tarea
void show_progress(int current, int total) {
    float progress = static_cast<float>(current) / total;
    int bar_width = 50;
    std::cout << "[";
    int pos = bar_width * progress;
    for (int i = 0; i < bar_width; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << " %\r";
    std::cout.flush();
}

// Normalizar los valores de píxeles de una imagen al rango [0, 1]
std::vector<float> normalize_image(const std::vector<float>& image) {
    std::vector<float> normalized(image.size());
    for (size_t i = 0; i < image.size(); ++i) {
        normalized[i] = image[i] / 255.0f;
    }
    return normalized;
}

// Codificar en one-hot una etiqueta en un vector de valores binarios
std::vector<int> one_hot_encode(int label, int num_classes) {
    std::vector<int> encoded(num_classes, 0);
    encoded[label] = 1;
    return encoded;
}

// Calcular la precisión de las etiquetas predichas en comparación con las etiquetas verdaderas
float calculate_accuracy(const std::vector<int>& predicted_labels, const std::vector<int>& true_labels) {
    int correct = 0;
    for (size_t i = 0; i < predicted_labels.size(); ++i) {
        if (predicted_labels[i] == true_labels[i]) {
            correct++;
        }
    }
    return static_cast<float>(correct) / predicted_labels.size();
}

// Guardar los resultados del entrenamiento (pérdidas y precisiones) en un archivo
void save_training_results(const std::string& filename, const std::vector<float>& losses, const std::vector<float>& accuracies) {
    std::ofstream file(filename);
    if (file.is_open()) {
        for (size_t i = 0; i < losses.size(); ++i) {
            file << losses[i] << "," << accuracies[i] << std::endl;
        }
        file.close();
    }
}

// Cargar los resultados del entrenamiento (pérdidas y precisiones) desde un archivo
void load_training_results(const std::string& filename, std::vector<float>& losses, std::vector<float>& accuracies) {
    std::ifstream file(filename);
    std::string line;
    if (file.is_open()) {
        while (std::getline(file, line)) {
            size_t comma_pos = line.find(',');
            if (comma_pos != std::string::npos) {
                losses.push_back(std::stof(line.substr(0, comma_pos)));
                accuracies.push_back(std::stof(line.substr(comma_pos + 1)));
            }
        }
        file.close();
    }
}