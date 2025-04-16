#include "FeatureExtractor.h"
#include <iostream> // For potential debug output

namespace sentiment
{

void FeatureExtractor::buildVocabulary(const std::vector<std::vector<std::string>>& tokenized_docs)
{
    vocabulary_.clear();
    size_t index = 0;
    for (const auto& doc_tokens : tokenized_docs) {
        for (const auto& token : doc_tokens) {
            if (vocabulary_.find(token) == vocabulary_.end()) {
                vocabulary_[token] = index++;
            }
        }
    }
    vocabulary_size_ = vocabulary_.size();
    // std::cout << "Built vocabulary with " << vocabulary_size_ << " unique words." << std::endl;
}

FeatureExtractor::FeatureVector FeatureExtractor::extractFeatures(
    const std::vector<std::string>& tokens) const
{
    // Initialize feature vector with zeros, size of the vocabulary
    FeatureVector feature_vec(vocabulary_size_, 0.0);

    for (const auto& token : tokens) {
        auto it = vocabulary_.find(token);
        if (it != vocabulary_.end()) {
            // Increment the count for the word if it's in the vocabulary
            size_t index = it->second;
            feature_vec[index] += 1.0;
        }
        // Words not in the vocabulary are ignored (as they were not seen during training)
    }
    return feature_vec;
}

size_t FeatureExtractor::getVocabularySize() const {
    return vocabulary_size_;
}

const FeatureExtractor::Vocabulary& FeatureExtractor::getVocabulary() const {
    return vocabulary_;
}


} // namespace sentiment
