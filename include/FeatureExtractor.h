#ifndef FEATUREEXTRACTOR_H
#define FEATUREEXTRACTOR_H

#include <vector>
#include <string>
#include <unordered_map>
#include <set> // For ordered vocabulary keys if needed

namespace sentiment
{

/**
 * @brief Extracts features from tokenized text using Bag-of-Words.
 */
class FeatureExtractor
{
public:
    // Type alias for feature vectors (using double for potential TF-IDF later)
    using FeatureVector = std::vector<double>;
    using Vocabulary = std::unordered_map<std::string, size_t>;

    /**
     * @brief Builds the vocabulary from a collection of tokenized documents.
     *
     * Assigns a unique index to each distinct token found in the training data.
     *
     * @param tokenized_docs A vector where each element is a document represented
     * as a vector of tokens.
     */
    void buildVocabulary(const std::vector<std::vector<std::string>>& tokenized_docs);

    /**
     * @brief Extracts a Bag-of-Words feature vector for a single tokenized document.
     *
     * Creates a vector where the i-th element represents the frequency count
     * of the i-th word in the vocabulary within the given document.
     *
     * @param tokens The tokenized document.
     * @return A FeatureVector representing the Bag-of-Words counts.
     */
    FeatureVector extractFeatures(const std::vector<std::string>& tokens) const;

    /**
     * @brief Gets the current vocabulary size.
     * @return The number of unique words in the vocabulary.
     */
    size_t getVocabularySize() const;

     /**
      * @brief Gets the vocabulary map (const reference).
      * @return A const reference to the internal vocabulary map.
      */
    const Vocabulary& getVocabulary() const;


private:
    Vocabulary vocabulary_;
    size_t vocabulary_size_ = 0;
};

} // namespace sentiment

#endif // FEATUREEXTRACTOR_H
