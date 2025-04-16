#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <random>
#include <chrono>

namespace sentiment {

/**
 * @brief Enumeration for sentiment labels
 */
enum class SentimentLabel {
    POSITIVE,
    NEGATIVE,
    NEUTRAL,
    UNKNOWN
};

/**
 * @brief Convert string sentiment to enum SentimentLabel
 * @param sentiment String representation of sentiment
 * @return Corresponding SentimentLabel enum value
 */
SentimentLabel stringToSentiment(const std::string& sentiment);

/**
 * @brief Convert enum SentimentLabel to string
 * @param label SentimentLabel enum value
 * @return String representation of sentiment
 */
std::string sentimentToString(SentimentLabel label);

/**
 * @brief Container for text data with sentiment label
 */
struct TextData {
    std::string text;
    SentimentLabel label;
};

/**
 * @brief Container for feature vector and label
 */
struct FeatureVector {
    std::vector<double> features;
    SentimentLabel label;
};

/**
 * @brief Evaluation metrics for classifier performance
 */
struct EvaluationMetrics {
    double accuracy;
    double precision;
    double recall;
    double f1Score;
};

/**
 * @brief Split a vector into two parts based on ratio
 * @param data Input vector to split
 * @param trainRatio Ratio of data to use for training (between 0 and 1)
 * @return Pair of vectors (train_data, validation_data)
 */
template<typename T>
std::pair<std::vector<T>, std::vector<T>> trainValidationSplit(
    const std::vector<T>& data,
    double trainRatio = 0.8
) {
    if (trainRatio <= 0.0 || trainRatio >= 1.0) {
        throw std::invalid_argument("Train ratio must be between 0 and 1");
    }

    std::vector<T> trainData;
    std::vector<T> validationData;

    size_t trainSize = static_cast<size_t>(data.size() * trainRatio);

    // Create a copy of the data
    std::vector<T> dataCopy = data;

    // Shuffle the data using a random engine
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::shuffle(dataCopy.begin(), dataCopy.end(), std::default_random_engine(seed));

    // Split the data
    trainData.insert(trainData.end(), dataCopy.begin(), dataCopy.begin() + trainSize);
    validationData.insert(validationData.end(), dataCopy.begin() + trainSize, dataCopy.end());

    return {trainData, validationData};
}

} // namespace sentiment

#endif // UTILS_H
