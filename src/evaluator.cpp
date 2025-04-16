#include "evaluator.h"
#include <iostream>
#include <iomanip>
#include <numeric>

namespace sentiment {

Evaluator::Evaluator(const Model& model) : model(model) {
}

EvaluationMetrics Evaluator::evaluate(const std::vector<FeatureVector>& validationData) {
    if (validationData.empty()) {
        std::cerr << "Error: Validation data is empty" << std::endl;
        return {0.0, 0.0, 0.0, 0.0};
    }

    // Clear previous results
    confusionMatrix.clear();

    // Initialize confusion matrix
    for (const auto& example : validationData) {
        if (confusionMatrix.find(example.label) == confusionMatrix.end()) {
            confusionMatrix[example.label] = {};
        }
    }

    // Evaluate model on validation data
    for (const auto& example : validationData) {
        SentimentLabel predicted = model.predict(example.features);
        confusionMatrix[example.label][predicted]++;
    }

    // Compute metrics
    double accuracyVal = computeAccuracy();
    double precisionSum = 0.0;
    double recallSum = 0.0;
    int labelCount = 0;

    // Calculate precision and recall for each label
    for (const auto& [label, _] : confusionMatrix) {
        double precision = computePrecision(label);
        double recall = computeRecall(label);

        if (!std::isnan(precision) && !std::isnan(recall)) {
            precisionSum += precision;
            recallSum += recall;
            labelCount++;
        }
    }

    // Calculate macro-averaged precision and recall
    double macroAvgPrecision = labelCount > 0 ? precisionSum / labelCount : 0.0;
    double macroAvgRecall = labelCount > 0 ? recallSum / labelCount : 0.0;

    // Calculate macro-averaged F1 score
    double f1ScoreVal = computeF1(macroAvgPrecision, macroAvgRecall);

    // Store metrics
    metrics = {
        accuracyVal,
        macroAvgPrecision,
        macroAvgRecall,
        f1ScoreVal
    };

    return metrics;
}

const std::unordered_map<SentimentLabel,
      std::unordered_map<SentimentLabel, int>>& Evaluator::getConfusionMatrix() const {
    return confusionMatrix;
}

void Evaluator::printResults() const {
    std::cout << "\n--- Evaluation Results for " << model.getName() << " ---\n";
    std::cout << "Accuracy:  " << std::fixed << std::setprecision(4) << metrics.accuracy * 100 << "%\n";
    std::cout << "Precision: " << std::fixed << std::setprecision(4) << metrics.precision * 100 << "%\n";
    std::cout << "Recall:    " << std::fixed << std::setprecision(4) << metrics.recall * 100 << "%\n";
    std::cout << "F1 Score:  " << std::fixed << std::setprecision(4) << metrics.f1Score * 100 << "%\n\n";

    // Print confusion matrix
    std::cout << "Confusion Matrix:\n";
    std::cout << "-----------------\n";
    std::cout << std::setw(10) << "Actual\\Pred";

    // Get all sentiment labels
    std::vector<SentimentLabel> labels;
    for (const auto& [label, _] : confusionMatrix) {
        labels.push_back(label);
    }

    // Sort labels for consistent output
    std::sort(labels.begin(), labels.end(), [](SentimentLabel a, SentimentLabel b) {
        return static_cast<int>(a) < static_cast<int>(b);
    });

    // Print column headers
    for (const auto& label : labels) {
        std::cout << std::setw(10) << sentimentToString(label);
    }
    std::cout << "\n";

    // Print each row
    for (const auto& trueLabel : labels) {
        std::cout << std::setw(10) << sentimentToString(trueLabel);

        for (const auto& predLabel : labels) {
            int count = 0;
            if (confusionMatrix.count(trueLabel) > 0 &&
                confusionMatrix.at(trueLabel).count(predLabel) > 0) {
                count = confusionMatrix.at(trueLabel).at(predLabel);
            }
            std::cout << std::setw(10) << count;
        }
        std::cout << "\n";
    }

    std::cout << std::endl;
}

double Evaluator::computePrecision(SentimentLabel label) const {
    int truePositives = 0;
    int falsePositives = 0;

    // Count true positives and false positives
    for (const auto& [trueLabel, predictions] : confusionMatrix) {
        for (const auto& [predLabel, count] : predictions) {
            if (predLabel == label) {
                if (trueLabel == label) {
                    truePositives += count;
                } else {
                    falsePositives += count;
                }
            }
        }
    }

    // Calculate precision
    if (truePositives + falsePositives > 0) {
        return static_cast<double>(truePositives) / (truePositives + falsePositives);
    }

    return 0.0;
}

double Evaluator::computeRecall(SentimentLabel label) const {
    int truePositives = 0;
    int falseNegatives = 0;

    // Count true positives
    if (confusionMatrix.count(label) > 0) {
        const auto& predictions = confusionMatrix.at(label);
        for (const auto& [predLabel, count] : predictions) {
            if (predLabel == label) {
                truePositives += count;
            } else {
                falseNegatives += count;
            }
        }
    }

    // Calculate recall
    if (truePositives + falseNegatives > 0) {
        return static_cast<double>(truePositives) / (truePositives + falseNegatives);
    }

    return 0.0;
}

double Evaluator::computeF1(double precision, double recall) const {
    if (precision + recall > 0.0) {
        return 2.0 * (precision * recall) / (precision + recall);
    }

    return 0.0;
}

double Evaluator::computeAccuracy() const {
    int correct = 0;
    int total = 0;

    // Count correct predictions and total predictions
    for (const auto& [trueLabel, predictions] : confusionMatrix) {
        for (const auto& [predLabel, count] : predictions) {
            if (trueLabel == predLabel) {
                correct += count;
            }
            total += count;
        }
    }

    // Calculate accuracy
    if (total > 0) {
        return static_cast<double>(correct) / total;
    }

    return 0.0;
}

} // namespace sentiment
