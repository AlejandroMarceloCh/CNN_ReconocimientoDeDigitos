#include "mnist_loader.h"
#include <iostream>
#include <fstream>
#include <vector>

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
