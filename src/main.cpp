#include <iostream>
#include <vector>
#include "mnist_loader.h"
#include "cnn.h"

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

int main() {
    CNN<float> cnn; // Asegúrate de que tu clase CNN esté bien implementada

    std::vector<std::vector<float>> images;
    
    // Ruta correcta hacia la carpeta data
    std::string train_images_path = "../data/train-images.idx3-ubyte";
    
    std::cout << "Intentando cargar imágenes desde: " << train_images_path << std::endl;
    
    loadMNISTImages(train_images_path, images);
    
    if (images.empty()) {
        std::cerr << "No se cargaron imágenes." << std::endl;
        return 1;
    }

    // Solo para mostrar que las imágenes se han cargado
    std::cout << "Se cargaron " << images.size() << " imágenes." << std::endl;

    // Mostrar las dimensiones de la primera imagen
    if (!images.empty()) {
        std::cout << "Dimensiones de la primera imagen (1D): " << images[0].size() << " píxeles." << std::endl;
    }
    
    // Convertir la primera imagen de 1D a 2D (28x28 píxeles)
    std::vector<std::vector<float>> mnist_image = reshape_image(images[0], 28, 28);

    // Hacer el forward pass con la imagen ya convertida a 2D
    auto output = cnn.forward(mnist_image);

    // Mostrar la salida
    std::cout << "Salida de la red para la imagen cargada:\n";
    for (const auto& val : output) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    return 0;
}
