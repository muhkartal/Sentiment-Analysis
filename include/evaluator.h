#ifndef EVALUATOR_H
#define EVALUATOR_H

#include <vector>
#include <unordered_map>
#include "model.h"
#include "utils.h"

namespace sentiment {

/**
 * @brief Class for evaluating model performance
 *
 * This class handles evaluating a trained model on validation data
 * and computing various performance metrics.
 */
class Evaluator {
public:
    /**
     * @brief Constructor
     * @param model Reference to a trained Model object
     */
    explicit Evaluator(const Model& model);

    /**
     * @brief Evaluate model on validation data
     *
     * Computes accuracy, precision, recall, and F1 score.
     *
     * @param validationData Vector of validation examples
     * @return EvaluationMetrics structure with computed metrics
     */
    EvaluationMetrics evaluate(const std::vector<FeatureVector>& validationData);

    /**
     * @brief Get confusion matrix
     *
     * Returns a map where key is true label and value is map of predicted label to count.
     *
     * @return Confusion matrix as nested unordered_map
     */
    const std::unordered_map<SentimentLabel,
          std::unordered_map<SentimentLabel, int>>& getConfusionMatrix() const;

    /**
     * @brief Print evaluation results
     *
     * Prints accuracy, precision, recall, F1 score, and confusion matrix.
     */
    void printResults() const;

private:
    const Model& model; ///< Reference to the model being evaluated
    EvaluationMetrics metrics; ///< Computed evaluation metrics

    // Confusion matrix: true_label -> predicted_label -> count
    std::unordered_map<SentimentLabel,
                      std::unordered_map<SentimentLabel, int>> confusionMatrix;

    /**
     * @brief Compute precision for a label
     * @param label Target sentiment label
     * @return Precision value
     */
    double computePrecision(SentimentLabel label) const;

    /**
     * @brief Compute recall for a label
     * @param label Target sentiment label
     * @return Recall value
     */
    double computeRecall(SentimentLabel label) const;

    /**
     * @brief Compute F1 score for a label
     * @param precision Precision value
     * @param recall Recall value
     * @return F1 score
     */
    double computeF1(double precision, double recall) const;

    /**
     * @brief Compute overall accuracy
     * @return Accuracy value
     */
    double computeAccuracy() const;
};

} // namespace sentiment

#endif // EVALUATOR_H
