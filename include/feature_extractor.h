#ifndef FEATURE_EXTRACTOR_H
#define FEATURE_EXTRACTOR_H

#include <string>
#include <vector>
#include <unordered_map>
#include "preprocessor.h"
#include "utils.h"

namespace sentiment {

/**
 * @brief Class for extracting features from text
 *
 * This class handles converting text to feature vectors
 * using Bag-of-Words (BoW) or TF-IDF representation.
 */
class FeatureExtractor {
public:
    /**
     * @brief Enum for feature extraction method
     */
    enum class Method {
        BAG_OF_WORDS, ///< Bag of Words (term frequency)
        TF_IDF        ///< Term Frequency-Inverse Document Frequency
    };

    /**
     * @brief Constructor
     * @param preprocessor Reference to a Preprocessor object
     * @param method Feature extraction method (BoW or TF-IDF)
     */
    explicit FeatureExtractor(
        const Preprocessor& preprocessor,
        Method method = Method::BAG_OF_WORDS
    );

    /**
     * @brief Build vocabulary from training data
     *
     * This method processes the training data to:
     * - Extract unique words
     * - Build the vocabulary mapping
     * - Calculate document frequencies for TF-IDF
     *
     * @param textData Vector of TextData to build vocabulary from
     * @param minFrequency Minimum frequency for a word to be included in vocabulary
     * @param maxVocabSize Maximum vocabulary size (0 for unlimited)
     */
    void buildVocabulary(
        const std::vector<TextData>& textData,
        int minFrequency = 2,
        size_t maxVocabSize = 5000
    );

    /**
     * @brief Convert text to feature vector
     * @param text Text to convert
     * @return Feature vector (sparse or dense based on method)
     */
    std::vector<double> extractFeatures(const std::string& text) const;

    /**
     * @brief Convert TextData to FeatureVector
     * @param textData TextData containing text and label
     * @return FeatureVector with features and label
     */
    FeatureVector transform(const TextData& textData) const;

    /**
     * @brief Convert a batch of TextData to FeatureVectors
     * @param textDataBatch Vector of TextData to transform
     * @return Vector of FeatureVectors
     */
    std::vector<FeatureVector> batchTransform(
        const std::vector<TextData>& textDataBatch
    ) const;

    /**
     * @brief Get the size of the vocabulary
     * @return Number of words in vocabulary
     */
    size_t getVocabularySize() const;

    /**
     * @brief Get the vocabulary map
     * @return Unordered map of word to index
     */
    const std::unordered_map<std::string, size_t>& getVocabulary() const;

    /**
     * @brief Get feature extraction method
     * @return Current method (BAG_OF_WORDS or TF_IDF)
     */
    Method getMethod() const;

private:
    const Preprocessor& preprocessor; ///< Reference to text preprocessor
    Method method; ///< Feature extraction method

    std::unordered_map<std::string, size_t> vocabulary; ///< Word to index mapping
    std::vector<double> documentFrequencies; ///< Document frequencies for TF-IDF
    size_t documentCount = 0; ///< Total document count for IDF calculation

    /**
     * @brief Calculate TF-IDF for a word in a document
     * @param termFrequency Term frequency in the document
     * @param wordIndex Index of the word in vocabulary
     * @return TF-IDF value
     */
    double calculateTfIdf(double termFrequency, size_t wordIndex) const;
};

} // namespace sentiment

#endif // FEATURE_EXTRACTOR_H
