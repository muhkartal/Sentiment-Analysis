#include "naive_bayes.h"
#include <cmath>
#include <limits>
#include <iostream>

namespace sentiment {

NaiveBayes::NaiveBayes(double alpha) : alpha(alpha) {
}

bool NaiveBayes::train(const std::vector<FeatureVector>& trainingData) {
    if (trainingData.empty()) {
        std::cerr << "Error: Training data is empty" << std::endl;
        return false;
    }

    // Initialize class counts
    std::unordered_map<SentimentLabel, int> classCounts;

    // Determine feature count from first example
    featureCount = trainingData[0].features.size();

    // Initialize likelihood structures
    logLikelihoods.clear();
    classTotals.clear();

    // Count class occurrences
    for (const auto& example : trainingData) {
        classCounts[example.label]++;

        // Initialize class-specific data structures if not already done
        if (logLikelihoods.find(example.label) == logLikelihoods.end()) {
            logLikelihoods[example.label] = std::vector<double>(featureCount, 0.0);
            classTotals[example.label] = 0.0;
        }

        // Sum feature values for each class
        for (size_t i = 0; i < example.features.size(); ++i) {
            logLikelihoods[example.label][i] += example.features[i];
            classTotals[example.label] += example.features[i];
        }
    }

    // Calculate class priors
    int totalExamples = trainingData.size();
    for (const auto& [label, count] : classCounts) {
        classPriors[label] = static_cast<double>(count) / totalExamples;
    }

    // Calculate log likelihoods with Laplace smoothing
    for (auto& [label, likelihoods] : logLikelihoods) {
        for (size_t i = 0; i < featureCount; ++i) {
            // Apply Laplace smoothing
            double smoothedProb = (likelihoods[i] + alpha) /
                                (classTotals[label] + alpha * featureCount);

            // Store log probability for numerical stability
            likelihoods[i] = std::log(smoothedProb);
        }
    }

    std::cout << "Trained Naive Bayes with "
              << trainingData.size() << " examples and "
              << featureCount << " features" << std::endl;

    trained = true;
    return true;
}

SentimentLabel NaiveBayes::predict(const std::vector<double>& features) const {
    if (!trained) {
        std::cerr << "Error: Model not trained" << std::endl;
        return SentimentLabel::UNKNOWN;
    }

    if (features.size() != featureCount) {
        std::cerr << "Error: Feature vector size mismatch. Expected "
                  << featureCount << ", got " << features.size() << std::endl;
        return SentimentLabel::UNKNOWN;
    }

    // Calculate log probabilities for each class
    double maxLogProb = -std::numeric_limits<double>::infinity();
    SentimentLabel bestLabel = SentimentLabel::UNKNOWN;

    for (const auto& [label, prior] : classPriors) {
        // Start with log of prior probability
        double logProb = std::log(prior);

        // Add log likelihoods for each feature
        for (size_t i = 0; i < features.size(); ++i) {
            if (features[i] > 0) {
                logProb += features[i] * logLikelihoods.at(label)[i];
            }
        }

        // Keep track of the most probable class
        if (logProb > maxLogProb) {
            maxLogProb = logProb;
            bestLabel = label;
        }
    }

    return bestLabel;
}

bool NaiveBayes::isTrained() const {
    return trained;
}

std::string NaiveBayes::getName() const {
    return "Naive Bayes";
}

} // namespace sentiment
