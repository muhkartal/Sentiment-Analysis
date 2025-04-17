#ifndef SENTIMENT_API_H
#define SENTIMENT_API_H

/**
 * @file sentiment_api.h
 * @brief Public API for the C++ Sentiment Analysis Engine
 *
 * This header provides the main entry points for using the sentiment
 * analysis library in external applications. It abstracts the underlying
 * implementation details and provides a simple interface for training,
 * evaluating, and using sentiment models.
 */

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include "utils.h"

namespace sentiment {

/**
 * @brief Configuration options for SentimentAnalyzer
 */
struct SentimentConfig {
    // Feature extraction options
    bool useStopWords = true;
    FeatureExtractor::Method featureMethod = FeatureExtractor::Method::BAG_OF_WORDS;
    int minWordFrequency = 2;
    size_t maxVocabularySize = 5000;

    // Model options
    double naiveBayesAlpha = 1.0;  // Laplace smoothing parameter

    // Training options
    double trainRatio = 0.8;  // Train/validation split ratio
};

/**
 * @brief Primary interface for sentiment analysis functionality
 *
 * This class provides a simplified API for the entire sentiment analysis
 * pipeline, encapsulating the underlying components and providing
 * methods for common tasks.
 */
class SentimentAnalyzer {
public:
    /**
     * @brief Constructor with optional configuration
     * @param config Configuration options
     */
    explicit SentimentAnalyzer(const SentimentConfig& config = SentimentConfig{});

    /**
     * @brief Destructor
     */
    ~SentimentAnalyzer();

    /**
     * @brief Load training data from a CSV file
     *
     * @param filePath Path to the CSV file
     * @param hasHeader Whether the CSV file has a header row
     * @param textColumn Index of the column containing text data
     * @param labelColumn Index of the column containing sentiment labels
     * @return true if loading was successful, false otherwise
     */
    bool loadTrainingData(
        const std::string& filePath,
        bool hasHeader = true,
        int textColumn = 0,
        int labelColumn = 1
    );

    /**
     * @brief Train the sentiment analysis model
     * @return true if training was successful, false otherwise
     */
    bool train();

    /**
     * @brief Evaluate model performance on validation data
     * @return Evaluation metrics structure
     */
    EvaluationMetrics evaluate();

    /**
     * @brief Predict sentiment for a piece of text
     * @param text Input text to analyze
     * @return Predicted sentiment label
     */
    SentimentLabel predict(const std::string& text) const;

    /**
     * @brief Predict sentiment with confidence scores
     *
     * @param text Input text to analyze
     * @return Map of sentiment labels to confidence scores (0-1)
     */
    std::unordered_map<SentimentLabel, double> predictWithConfidence(
        const std::string& text
    ) const;

    /**
     * @brief Save the trained model to a file
     * @param filePath Path to save the model
     * @return true if saving was successful, false otherwise
     */
    bool saveModel(const std::string& filePath) const;

    /**
     * @brief Load a pre-trained model from a file
     * @param filePath Path to the model file
     * @return true if loading was successful, false otherwise
     */
    bool loadModel(const std::string& filePath);

    /**
     * @brief Get evaluation metrics from the last evaluation
     * @return Evaluation metrics structure
     */
    const EvaluationMetrics& getMetrics() const;

    /**
     * @brief Get confusion matrix from the last evaluation
     * @return Confusion matrix as nested map
     */
    const std::unordered_map<SentimentLabel,
          std::unordered_map<SentimentLabel, int>>& getConfusionMatrix() const;

private:
    class Impl;
    std::unique_ptr<Impl> pImpl;  // Private implementation (PIMPL idiom)
};

} // namespace sentiment

#endif // SENTIMENT_API_H
