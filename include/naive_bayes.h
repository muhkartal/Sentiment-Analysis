#ifndef NAIVE_BAYES_H
#define NAIVE_BAYES_H

#include <vector>
#include <unordered_map>
#include "model.h"

namespace sentiment {

/**
 * @brief Multinomial Naive Bayes classifier for sentiment analysis
 *
 * Implements a Multinomial Naive Bayes classifier which is well-suited
 * for text classification, especially with bag-of-words features.
 */
class NaiveBayes : public Model {
public:
    /**
     * @brief Constructor
     * @param alpha Laplace smoothing parameter (default: 1.0)
     */
    explicit NaiveBayes(double alpha = 1.0);

    /**
     * @brief Train the Naive Bayes model
     *
     * Calculates class priors and conditional probabilities
     * for each feature given each class.
     *
     * @param trainingData Vector of training examples (feature vectors with labels)
     * @return true if training was successful, false otherwise
     */
    bool train(const std::vector<FeatureVector>& trainingData) override;

    /**
     * @brief Predict sentiment for a feature vector
     *
     * Uses the Naive Bayes formula:
     * P(class|features) ∝ P(class) * ∏ P(feature_i|class)
     *
     * @param features Input feature vector
     * @return Predicted sentiment label
     */
    SentimentLabel predict(const std::vector<double>& features) const override;

    /**
     * @brief Check if the model is trained
     * @return true if the model is trained, false otherwise
     */
    bool isTrained() const override;

    /**
     * @brief Get model name
     * @return String "Naive Bayes"
     */
    std::string getName() const override;

private:
    double alpha; ///< Laplace smoothing parameter
    bool trained = false; ///< Whether the model has been trained
    size_t featureCount = 0; ///< Number of features

    // Priors: P(class)
    std::unordered_map<SentimentLabel, double> classPriors;

    // Likelihood: log(P(feature|class))
    std::unordered_map<SentimentLabel, std::vector<double>> logLikelihoods;

    // Feature sum for each class
    std::unordered_map<SentimentLabel, double> classTotals;
};

} // namespace sentiment

#endif // NAIVE_BAYES_H
