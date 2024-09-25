#include "mnist_loader.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>

// Función para cargar imágenes desde el archivo MNIST
void loadMNISTImages(const std::string &filename, std::vector<std::vector<float>> &images) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "No se pudo abrir el archivo " << filename << std::endl;
        return;
    }

    uint32_t magicNumber;
    uint32_t numImages;
    uint32_t numRows;
    uint32_t numCols;

    // Leer encabezado
    file.read(reinterpret_cast<char*>(&magicNumber), sizeof(magicNumber));
    magicNumber = __builtin_bswap32(magicNumber); // Convertir de big-endian a little-endian
    file.read(reinterpret_cast<char*>(&numImages), sizeof(numImages));
    numImages = __builtin_bswap32(numImages);
    file.read(reinterpret_cast<char*>(&numRows), sizeof(numRows));
    file.read(reinterpret_cast<char*>(&numCols), sizeof(numCols));
    numRows = __builtin_bswap32(numRows);
    numCols = __builtin_bswap32(numCols);

    images.resize(numImages, std::vector<float>(numRows * numCols));
    for (uint32_t i = 0; i < numImages; ++i) {
        std::vector<uint8_t> tempImage(numRows * numCols);
        file.read(reinterpret_cast<char*>(tempImage.data()), numRows * numCols);

        // Normalización: Convertimos los valores a float y los dividimos entre 255
        for (uint32_t j = 0; j < tempImage.size(); ++j) {
            images[i][j] = static_cast<float>(tempImage[j]) / 255.0f;
        }
    }

    file.close();
}

// Función para cargar las etiquetas del archivo MNIST
void loadMNISTLabels(const std::string& path, std::vector<int>& labels) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error al abrir el archivo de etiquetas: " << path << std::endl;
        return;
    }

    // Leer el encabezado del archivo
    int magic_number;
    int number_of_labels;
    file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
    file.read(reinterpret_cast<char*>(&number_of_labels), sizeof(number_of_labels));

    // Convertir de big-endian a little-endian
    magic_number = __builtin_bswap32(magic_number);
    number_of_labels = __builtin_bswap32(number_of_labels);

    // Leer las etiquetas
    labels.resize(number_of_labels);
    for (int i = 0; i < number_of_labels; ++i) {
        unsigned char label;
        file.read(reinterpret_cast<char*>(&label), sizeof(label));
        labels[i] = static_cast<int>(label);
    }

    file.close();
}
