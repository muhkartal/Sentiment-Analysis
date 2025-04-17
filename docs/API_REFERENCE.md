# C++ Sentiment Analysis Engine API Reference

This document provides detailed information about using the Sentiment Analysis Engine API.

## Table of Contents

-  [Overview](#overview)
-  [SentimentAnalyzer Class](#sentimentanalyzer-class)
-  [SentimentConfig Structure](#sentimentconfig-structure)
-  [Enumerations](#enumerations)
-  [Utilities](#utilities)
-  [Error Handling](#error-handling)
-  [Usage Examples](#usage-examples)

## Overview

The Sentiment Analysis Engine provides a high-level API for:

1. Loading and preparing training data
2. Building and training sentiment classification models
3. Evaluating model performance
4. Using trained models to predict sentiment of new text

The API is designed to be simple to use while providing access to advanced features when needed.

## SentimentAnalyzer Class

`SentimentAnalyzer` is the main entry point for the API.

### Constructor

```cpp
explicit SentimentAnalyzer(const SentimentConfig& config = SentimentConfig{});
```

Creates a new sentiment analyzer with optional configuration settings.

### Methods

#### Data Loading

```cpp
bool loadTrainingData(
    const std::string& filePath,
    bool hasHeader = true,
    int textColumn = 0,
    int labelColumn = 1
);
```

Loads labeled text data from a CSV file.

-  **Parameters:**
   -  `filePath`: Path to the CSV file
   -  `hasHeader`: Whether the CSV file has a header row
   -  `textColumn`: Index of the column containing text data
   -  `labelColumn`: Index of the column containing sentiment labels
-  **Returns:** `true` if loading was successful, `false` otherwise

#### Training

```cpp
bool train();
```

Trains the model using previously loaded data.

-  **Returns:** `true` if training was successful, `false` otherwise

#### Evaluation

```cpp
EvaluationMetrics evaluate();
```

Evaluates model performance on validation data.

-  **Returns:** Structure containing accuracy, precision, recall, and F1 score

#### Prediction

```cpp
SentimentLabel predict(const std::string& text) const;
```

Predicts sentiment for a piece of text.

-  **Parameters:**
   -  `text`: Input text to analyze
-  **Returns:** Predicted sentiment label (POSITIVE, NEGATIVE, NEUTRAL, or UNKNOWN)

```cpp
std::unordered_map<SentimentLabel, double> predictWithConfidence(
    const std::string& text
) const;
```

Predicts sentiment with confidence scores.

-  **Parameters:**
   -  `text`: Input text to analyze
-  **Returns:** Map of sentiment labels to confidence scores (0-1)

#### Model Persistence

```cpp
bool saveModel(const std::string& filePath) const;
```

Saves the trained model to a file.

-  **Parameters:**
   -  `filePath`: Path to save the model
-  **Returns:** `true` if saving was successful, `false` otherwise

```cpp
bool loadModel(const std::string& filePath);
```

Loads a pre-trained model from a file.

-  **Parameters:**
   -  `filePath`: Path to the model file
-  **Returns:** `true` if loading was successful, `false` otherwise

#### Results Retrieval

```cpp
const EvaluationMetrics& getMetrics() const;
```

Gets evaluation metrics from the last evaluation.

-  **Returns:** Evaluation metrics structure

```cpp
const std::unordered_map<SentimentLabel,
      std::unordered_map<SentimentLabel, int>>& getConfusionMatrix() const;
```

Gets confusion matrix from the last evaluation.

-  **Returns:** Confusion matrix as nested map

## SentimentConfig Structure

Configuration options for the `SentimentAnalyzer`:

```cpp
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
```

### Options

-  **useStopWords**: Whether to remove common stop words during preprocessing
-  **featureMethod**: Method for feature extraction (BAG_OF_WORDS or TF_IDF)
-  **minWordFrequency**: Minimum frequency for a word to be included in vocabulary
-  **maxVocabularySize**: Maximum vocabulary size (0 for unlimited)
-  **naiveBayesAlpha**: Laplace smoothing parameter for Naive Bayes
-  **trainRatio**: Portion of data to use for training vs. validation

## Enumerations

### SentimentLabel

```cpp
enum class SentimentLabel {
    POSITIVE,
    NEGATIVE,
    NEUTRAL,
    UNKNOWN
};
```

Represents the sentiment classification of text.

### FeatureExtractor::Method

```cpp
enum class FeatureExtractor::Method {
    BAG_OF_WORDS,  // Term frequency counts
    TF_IDF         // Term Frequency-Inverse Document Frequency
};
```

Methods for converting text to numerical feature vectors.

## Utilities

### sentimentToString

```cpp
std::string sentimentToString(SentimentLabel label);
```

Converts a `SentimentLabel` enum to a readable string.

-  **Parameters:**
   -  `label`: SentimentLabel enum value
-  **Returns:** String representation ("positive", "negative", "neutral", or "unknown")

### stringToSentiment

```cpp
SentimentLabel stringToSentiment(const std::string& sentiment);
```

Converts a string to a `SentimentLabel` enum.

-  **Parameters:**
   -  `sentiment`: String representation of sentiment
-  **Returns:** Corresponding `SentimentLabel` enum value

## Error Handling

The API uses return values to indicate success or failure rather than exceptions. Methods that can fail return a boolean value. Error messages are logged to `std::cerr`.

To enable more detailed error information, check return values and examine the error output.

## Usage Examples

### Basic Usage

```cpp
#include "sentiment_api.h"
#include <iostream>

int main() {
    // Create analyzer with default settings
    sentiment::SentimentAnalyzer analyzer;

    // Load training data
    if (!analyzer.loadTrainingData("data/sample_data.csv")) {
        std::cerr << "Failed to load data" << std::endl;
        return 1;
    }

    // Train model
    if (!analyzer.train()) {
        std::cerr << "Failed to train model" << std::endl;
        return 1;
    }

    // Predict sentiment
    sentiment::SentimentLabel label = analyzer.predict("I love this product!");
    std::cout << "Sentiment: " << sentiment::sentimentToString(label) << std::endl;

    return 0;
}
```

### Advanced Configuration

```cpp
#include "sentiment_api.h"

int main() {
    // Configure analyzer with custom settings
    sentiment::SentimentConfig config;
    config.useStopWords = true;
    config.featureMethod = sentiment::FeatureExtractor::Method::TF_IDF;
    config.minWordFrequency = 3;
    config.maxVocabularySize = 10000;
    config.naiveBayesAlpha = 0.5;

    sentiment::SentimentAnalyzer analyzer(config);

    // Use analyzer as in basic example...

    return 0;
}
```

### Model Persistence

```cpp
#include "sentiment_api.h"

void trainAndSave() {
    sentiment::SentimentAnalyzer analyzer;
    analyzer.loadTrainingData("data/training.csv");
    analyzer.train();
    analyzer.saveModel("models/sentiment_model.bin");
}

void loadAndUse() {
    sentiment::SentimentAnalyzer analyzer;
    analyzer.loadModel("models/sentiment_model.bin");

    // Use the model for prediction
    sentiment::SentimentLabel label = analyzer.predict("Great service!");
}
```

### Detailed Evaluation

```cpp
#include "sentiment_api.h"
#include <iostream>
#include <iomanip>

int main() {
    sentiment::SentimentAnalyzer analyzer;
    analyzer.loadTrainingData("data/data.csv");
    analyzer.train();

    // Evaluate model
    sentiment::EvaluationMetrics metrics = analyzer.evaluate();

    // Print metrics
    std::cout << "Accuracy:  " << std::fixed << std::setprecision(4)
              << metrics.accuracy * 100 << "%" << std::endl;
    std::cout << "Precision: " << metrics.precision * 100 << "%" << std::endl;
    std::cout << "Recall:    " << metrics.recall * 100 << "%" << std::endl;
    std::cout << "F1 Score:  " << metrics.f1Score * 100 << "%" << std::endl;

    // Get confidence scores for a prediction
    auto confidences = analyzer.predictWithConfidence("Good experience overall.");
    for (const auto& [label, score] : confidences) {
        std::cout << sentiment::sentimentToString(label) << ": "
                  << score * 100 << "%" << std::endl;
    }

    return 0;
}
```
