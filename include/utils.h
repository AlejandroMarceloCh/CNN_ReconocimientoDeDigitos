#ifndef UTILS_H
#define UTILS_H

#include "cnn.h"
#include <vector>
#include <string>

// Prototipos de funciones
std::vector<std::vector<float>> reshape_image(const std::vector<float>& image_1d, int width, int height);
float evaluate(CNN<float>& cnn, const std::vector<std::vector<float>>& test_images, const std::vector<int>& test_labels);
float cross_entropy_loss(const std::vector<float>& predicted, const std::vector<int>& labels);
void show_progress(int current, int total);
std::vector<float> normalize_image(const std::vector<float>& image);
std::vector<int> one_hot_encode(int label, int num_classes);
float calculate_accuracy(const std::vector<int>& predicted_labels, const std::vector<int>& true_labels);
void save_training_results(const std::string& filename, const std::vector<float>& losses, const std::vector<float>& accuracies);
void load_training_results(const std::string& filename, std::vector<float>& losses, std::vector<float>& accuracies);

#endif
