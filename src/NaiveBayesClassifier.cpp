#include "NaiveBayesClassifier.h"
#include <cmath>     // For std::log
#include <limits>    // For std::numeric_limits
#include <iostream>  // For debug/info messages
#include <set>       // To find unique classes easily

namespace sentiment
{

NaiveBayesClassifier::NaiveBayesClassifier(double alpha) : alpha_(alpha), vocabulary_size_(0) {
    if (alpha_ <= 0) {
        // Prevent non-positive smoothing values which cause issues
        std::cerr << "Warning: Laplace smoothing factor alpha must be positive. Setting to 1.0." << std::endl;
        alpha_ = 1.0;
    }
}

void NaiveBayesClassifier::train(const std::vector<FeatureVector>& features,
                                 const std::vector<std::string>& labels,
                                 size_t vocab_size)
{
    vocabulary_size_ = vocab_size;
    total_docs_trained_ = features.size();
    if (total_docs_trained_ == 0 || total_docs_trained_ != labels.size()) {
        throw std::runtime_error("NaiveBayes Error: Invalid training data size.");
    }
    if (vocabulary_size_ == 0) {
         throw std::runtime_error("NaiveBayes Error: Vocabulary size cannot be zero.");
    }


    // --- Step 1: Calculate Class Priors and identify unique classes ---
    std::unordered_map<std::string, int> class_counts;
    std::set<std::string> unique_classes; // Use set to automatically get unique labels
    for (const auto& label : labels) {
        class_counts[label]++;
        unique_classes.insert(label);
    }

    classes_.assign(unique_classes.begin(), unique_classes.end()); // Store unique classes

    for (const auto& cls : classes_) {
        // Calculate log prior: log(P(class)) = log(count(class) / total_docs)
        class_priors_[cls] = std::log(static_cast<double>(class_counts[cls]) / total_docs_trained_);
        total_words_in_class_[cls] = 0.0; // Initialize word counts for each class
        log_likelihoods_[cls] = {}; // Initialize likelihood map for the class
    }


    // --- Step 2: Calculate Word Counts per Class ---
    // Map to store count of each word (feature index) within each class
    std::unordered_map<std::string, std::unordered_map<size_t, double>> word_counts_per_class;

    for (size_t i = 0; i < features.size(); ++i) {
        const auto& feature_vec = features[i];
        const std::string& label = labels[i];

        // Iterate through the feature vector (representing word counts in doc i)
        for (size_t word_index = 0; word_index < feature_vec.size(); ++word_index) {
            double count = feature_vec[word_index]; // Get the count of this word in the doc
            if (count > 0) {
                word_counts_per_class[label][word_index] += count;
                total_words_in_class_[label] += count;
            }
        }
    }

    // --- Step 3: Calculate Log Likelihoods with Laplace Smoothing ---
    // Calculate the denominator term once for each class (total words + alpha * V)
    std::unordered_map<std::string, double> class_denominator;
     for (const auto& cls : classes_) {
        class_denominator[cls] = total_words_in_class_[cls] + alpha_ * vocabulary_size_;
        // Calculate default log likelihood for unseen words in this class
        // log( (0 + alpha) / (total_words_in_class + alpha * V) )
        default_log_likelihood_[cls] = std::log(alpha_ / class_denominator[cls]);
     }

    // Calculate log likelihood for each word seen in training for each class
    for(const auto& pair_class_counts : word_counts_per_class) {
        const std::string& cls = pair_class_counts.first;
        const auto& word_counts = pair_class_counts.second; // map: word_index -> count

        for (const auto& pair_word_count : word_counts) {
            size_t word_index = pair_word_count.first;
            double count = pair_word_count.second;

            // log P(word | class) = log( (count(word, class) + alpha) / (total_words_in_class + alpha * V) )
            log_likelihoods_[cls][word_index] = std::log((count + alpha_) / class_denominator[cls]);
        }
    }
    // Note: Words never seen for a class will implicitly use the default_log_likelihood_
    // when predicted later, as they won't be found in log_likelihoods_[cls].
}


std::string NaiveBayesClassifier::predict(const FeatureVector& feature_vec) const
{
    if (classes_.empty()) {
        throw std::runtime_error("NaiveBayes Error: Model has not been trained.");
    }
     if (feature_vec.size() != vocabulary_size_) {
         // This shouldn't happen if FeatureExtractor is used correctly post-training
         throw std::runtime_error("NaiveBayes Error: Feature vector size mismatch with vocabulary size during prediction.");
     }


    std::string best_class;
    double max_log_prob = -std::numeric_limits<double>::infinity();

    // Calculate the posterior log probability for each class
    for (const auto& cls : classes_) {
        // Start with the log prior probability P(class)
        double current_log_prob = class_priors_.at(cls);

        // Add the log likelihoods for words present in the input feature vector
        // Sum over all words i in the vocabulary: count(word_i) * log(P(word_i | class))
        for (size_t word_index = 0; word_index < feature_vec.size(); ++word_index) {
            double word_count = feature_vec[word_index];
            if (word_count > 0) { // Only consider words present in the input document
                double log_lik;
                // Find the pre-calculated log likelihood for this word in this class
                auto it_class_likelihoods = log_likelihoods_.find(cls);
                if(it_class_likelihoods != log_likelihoods_.end()) {
                    auto it_word_lik = it_class_likelihoods->second.find(word_index);
                    if (it_word_lik != it_class_likelihoods->second.end()) {
                        // Word was seen in this class during training
                        log_lik = it_word_lik->second;
                    } else {
                        // Word was NOT seen in this specific class during training (but exists in vocab)
                        // Use the default smoothed likelihood for unseen words in this class
                        log_lik = default_log_likelihood_.at(cls);
                    }
                } else {
                     // Should not happen if training was correct, but defensively use default
                    log_lik = default_log_likelihood_.at(cls);
                }

                current_log_prob += word_count * log_lik; // Multiply count by log probability
            }
        }

        // Update best class if current class has higher probability
        if (current_log_prob > max_log_prob) {
            max_log_prob = current_log_prob;
            best_class = cls;
        }
    }

    // If somehow no class was assigned (e.g., all probabilities were -inf, though unlikely with smoothing)
    // return the first class as a fallback.
    if (best_class.empty() && !classes_.empty()) {
        return classes_[0];
    }

    return best_class;
}

const std::vector<std::string>& NaiveBayesClassifier::getClasses() const {
    return classes_;
}


} // namespace sentiment
