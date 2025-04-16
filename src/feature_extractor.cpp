#include "feature_extractor.h"
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <iostream>
#include <limits>

namespace sentiment {

FeatureExtractor::FeatureExtractor(
    const Preprocessor& preprocessor,
    Method method
) : preprocessor(preprocessor), method(method) {
}

void FeatureExtractor::buildVocabulary(
    const std::vector<TextData>& textData,
    int minFrequency,
    size_t maxVocabSize
) {
    // Count word frequencies across all documents
    std::unordered_map<std::string, int> wordFrequencies;
    std::unordered_map<std::string, int> documentOccurrences;

    documentCount = textData.size();

    // First pass: count word frequencies and document occurrences
    for (const auto& data : textData) {
        // Get tokens for this document
        std::vector<std::string> tokens = preprocessor.preprocess(data.text);

        // Keep track of words seen in this document
        std::unordered_set<std::string> uniqueWordsInDoc;

        // Count token frequencies
        for (const auto& token : tokens) {
            wordFrequencies[token]++;
            uniqueWordsInDoc.insert(token);
        }

        // Update document occurrences
        for (const auto& word : uniqueWordsInDoc) {
            documentOccurrences[word]++;
        }
    }

    // Filter words by minimum frequency
    std::vector<std::pair<std::string, int>> filteredWords;
    for (const auto& [word, count] : wordFrequencies) {
        if (count >= minFrequency) {
            filteredWords.push_back({word, count});
        }
    }

    // Sort by frequency (descending)
    std::sort(filteredWords.begin(), filteredWords.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    // Limit vocabulary size if needed
    if (maxVocabSize > 0 && filteredWords.size() > maxVocabSize) {
        filteredWords.resize(maxVocabSize);
    }

    // Build vocabulary mapping
    vocabulary.clear();
    size_t index = 0;
    for (const auto& [word, _] : filteredWords) {
        vocabulary[word] = index++;
    }

    // If using TF-IDF, prepare document frequencies
    if (method == Method::TF_IDF) {
        documentFrequencies.resize(vocabulary.size(), 0.0);

        for (const auto& [word, index] : vocabulary) {
            if (documentOccurrences.find(word) != documentOccurrences.end()) {
                documentFrequencies[index] = documentOccurrences[word];
            }
        }
    }

    std::cout << "Vocabulary built with " << vocabulary.size() << " words" << std::endl;
}

std::vector<double> FeatureExtractor::extractFeatures(const std::string& text) const {
    // Preprocess the text
    std::vector<std::string> tokens = preprocessor.preprocess(text);

    // Initialize feature vector with zeros
    std::vector<double> features(vocabulary.size(), 0.0);

    // Count token frequencies
    for (const auto& token : tokens) {
        auto it = vocabulary.find(token);
        if (it != vocabulary.end()) {
            features[it->second] += 1.0;
        }
    }

    // If using TF-IDF, apply the transformation
    if (method == Method::TF_IDF) {
        for (size_t i = 0; i < features.size(); ++i) {
            if (features[i] > 0) {
                features[i] = calculateTfIdf(features[i], i);
            }
        }
    }

    return features;
}

FeatureVector FeatureExtractor::transform(const TextData& textData) const {
    return {extractFeatures(textData.text), textData.label};
}

std::vector<FeatureVector> FeatureExtractor::batchTransform(
    const std::vector<TextData>& textDataBatch
) const {
    std::vector<FeatureVector> featureVectors;
    featureVectors.reserve(textDataBatch.size());

    for (const auto& data : textDataBatch) {
        featureVectors.push_back(transform(data));
    }

    return featureVectors;
}

size_t FeatureExtractor::getVocabularySize() const {
    return vocabulary.size();
}

const std::unordered_map<std::string, size_t>& FeatureExtractor::getVocabulary() const {
    return vocabulary;
}

FeatureExtractor::Method FeatureExtractor::getMethod() const {
    return method;
}

double FeatureExtractor::calculateTfIdf(double termFrequency, size_t wordIndex) const {
    if (documentCount == 0 || wordIndex >= documentFrequencies.size()) {
        return 0.0;
    }

    double docFreq = documentFrequencies[wordIndex];
    if (docFreq == 0.0) {
        return 0.0;
    }

    // Calculate TF-IDF: tf * log(N/df)
    double tf = termFrequency;
    double idf = std::log(static_cast<double>(documentCount) / docFreq);

    return tf * idf;
}

} // namespace sentiment
