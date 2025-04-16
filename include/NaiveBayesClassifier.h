#ifndef NAIVEBAYESCLASSIFIER_H
#define NAIVEBAYESCLASSIFIER_H

#include "FeatureExtractor.h" // For FeatureVector type
#include <vector>
#include <string>
#include <unordered_map>
#include <map> // For potentially ordered class outputs

namespace sentiment
{

/**
 * @brief Implements a Multinomial Naive Bayes classifier for text sentiment.
 */
class NaiveBayesClassifier
{
public:
    // Type alias from FeatureExtractor
    using FeatureVector = sentiment::FeatureExtractor::FeatureVector;

    /**
     * @brief Constructor.
     * @param alpha Laplace smoothing factor (defaults to 1.0).
     */
    explicit NaiveBayesClassifier(double alpha = 1.0);

    /**
     * @brief Trains the Naive Bayes model.
     *
     * Calculates class priors and word log-likelihoods based on the training data.
     *
     * @param features A vector of feature vectors for the training documents.
     * @param labels A vector of corresponding labels for the training documents.
     * @param vocab_size The total size of the vocabulary used to generate features.
     */
    void train(const std::vector<FeatureVector>& features,
               const std::vector<std::string>& labels,
               size_t vocab_size);

    /**
     * @brief Predicts the sentiment label for a given feature vector.
     * @param feature_vec The feature vector of the document to classify.
     * @return The predicted sentiment label (string).
     */
    std::string predict(const FeatureVector& feature_vec) const;

    /**
     * @brief Gets the set of classes the model was trained on.
     * @return A const reference to the set of class labels.
     */
    const std::vector<std::string>& getClasses() const;


private:
    double alpha_; // Laplace smoothing factor
    size_t vocabulary_size_;
    size_t total_docs_trained_ = 0;

    std::vector<std::string> classes_; // Store unique classes encountered during training
    std::unordered_map<std::string, double> class_priors_; // Log prior probability P(class)
    // Stores log likelihood P(word_i | class)
    // Outer map: class label -> Inner map: feature index (word index) -> log likelihood
    std::unordered_map<std::string, std::unordered_map<size_t, double>> log_likelihoods_;
    // Store total word counts per class (needed for likelihood calculation)
    std::unordered_map<std::string, double> total_words_in_class_;
    // Default log likelihood for words unseen in a class during training
    std::unordered_map<std::string, double> default_log_likelihood_;
};

} // namespace sentiment

#endif // NAIVEBAYESCLASSIFIER_H
