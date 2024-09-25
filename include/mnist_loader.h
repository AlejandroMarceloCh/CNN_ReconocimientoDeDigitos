#ifndef MNIST_LOADER_H
#define MNIST_LOADER_H

#include <vector>
#include <string>
#include <cstdint>

void loadMNISTImages(const std::string &filename, std::vector<std::vector<float>> &images);
void loadMNISTLabels(const std::string &filename, std::vector<int> &labels);


#endif // MNIST_LOADER_H
