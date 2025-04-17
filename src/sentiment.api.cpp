#include "sentiment_api.h"
#include "data_loader.h"
#include "preprocessor.h"
#include "feature_extractor.h"
#include "naive_bayes.h"
#include "evaluator.h"
#include <fstream>
#include <iostream>
#include <cmath>

namespace sentiment {

// Private implementation class (PIMPL idiom)
class SentimentAnalyzer::Impl {
public:
    SentimentConfig config;
    DataLoader dataLoader;
    Preprocessor preprocessor;
    FeatureExtractor featureExtractor;
    NaiveBayes model;
    Evaluator* evaluator = nullptr;

    std::vector<TextData> trainData;
    std::vector<TextData> validData;
    std::vector<FeatureVector> trainFeatures;
    std::vector<FeatureVector> validFeatures;

    EvaluationMetrics metrics;
    bool isTrained = false;

    Impl(const SentimentConfig& conf)
        : config(conf),
          preprocessor(conf.useStopWords),
          featureExtractor(preprocessor, conf.featureMethod),
          model(conf.naiveBayesAlpha) {
    }

    ~Impl() {
        delete evaluator;
    }
};

SentimentAnalyzer::SentimentAnalyzer(const SentimentConfig& config)
    : pImpl(std::make_unique<Impl>(config)) {
}

SentimentAnalyzer::~SentimentAnalyzer() = default;

bool SentimentAnalyzer::loadTrainingData(
    const std::string& filePath,
    bool hasHeader,
    int textColumn,
    int labelColumn
) {
    bool success = pImpl->dataLoader.loadFromCSV(filePath, hasHeader, textColumn, labelColumn);

    if (success) {
        // Split data into training and validation sets
        auto [train, valid] = pImpl->dataLoader.splitTrainValidation(pImpl->config.trainRatio);
        pImpl->trainData = std::move(train);
        pImpl->validData = std::move(valid);

        std::cout << "Loaded " << pImpl->dataLoader.getData().size() << " examples" << std::endl;
        std::cout << "Split into " << pImpl->trainData.size() << " training and "
                  << pImpl->validData.size() << " validation examples" << std::endl;
    }

    return success;
}

bool SentimentAnalyzer::train() {
    if (pImpl->trainData.empty()) {
        std::cerr << "Error: No training data loaded" << std::endl;
        return false;
    }

    // Build vocabulary from training data
    pImpl->featureExtractor.buildVocabulary(
        pImpl->trainData,
        pImpl->config.minWordFrequency,
        pImpl->config.maxVocabularySize
    );

    // Transform training data to feature vectors
    pImpl->trainFeatures = pImpl->featureExtractor.batchTransform(pImpl->trainData);

    // Transform validation data to feature vectors
    pImpl->validFeatures = pImpl->featureExtractor.batchTransform(pImpl->validData);

    // Train the model
    bool success = pImpl->model.train(pImpl->trainFeatures);
    pImpl->isTrained = success;

    return success;
}

EvaluationMetrics SentimentAnalyzer::evaluate() {
    if (!pImpl->isTrained) {
        std::cerr << "Error: Model not trained" << std::endl;
        return EvaluationMetrics{};
    }

    // Initialize evaluator if not already done
    if (!pImpl->evaluator) {
        pImpl->evaluator = new Evaluator(pImpl->model);
    }

    // Evaluate on validation data
    pImpl->metrics = pImpl->evaluator->evaluate(pImpl->validFeatures);

    return pImpl->metrics;
}

SentimentLabel SentimentAnalyzer::predict(const std::string& text) const {
    if (!pImpl->isTrained) {
        std::cerr << "Error: Model not trained" << std::endl;
        return SentimentLabel::UNKNOWN;
    }

    // Extract features from text
    std::vector<double> features = pImpl->featureExtractor.extractFeatures(text);

    // Predict sentiment
    return pImpl->model.predict(features);
}

std::unordered_map<SentimentLabel, double> SentimentAnalyzer::predictWithConfidence(
    const std::string& text
) const {
    // This is a simplified implementation that could be extended
    // in the future with actual confidence scores from model internals

    std::unordered_map<SentimentLabel, double> confidences;

    // For now, assign 1.0 to the predicted label and 0.0 to others
    SentimentLabel predictedLabel = predict(text);

    confidences[SentimentLabel::POSITIVE] = (predictedLabel == SentimentLabel::POSITIVE) ? 1.0 : 0.0;
    confidences[SentimentLabel::NEGATIVE] = (predictedLabel == SentimentLabel::NEGATIVE) ? 1.0 : 0.0;
    confidences[SentimentLabel::NEUTRAL] = (predictedLabel == SentimentLabel::NEUTRAL) ? 1.0 : 0.0;

    return confidences;
}

bool SentimentAnalyzer::saveModel(const std::string& filePath) const {
    if (!pImpl->isTrained) {
        std::cerr << "Error: Cannot save untrained model" << std::endl;
        return false;
    }

    // TODO: Implement model serialization
    std::cerr << "Model saving not yet implemented" << std::endl;
    return false;
}

bool SentimentAnalyzer::loadModel(const std::string& filePath) {
    // TODO: Implement model deserialization
    std::cerr << "Model loading not yet implemented" << std::endl;
    return false;
}

const EvaluationMetrics& SentimentAnalyzer::getMetrics() const {
    return pImpl->metrics;
}

const std::unordered_map<SentimentLabel,
      std::unordered_map<SentimentLabel, int>>& SentimentAnalyzer::getConfusionMatrix() const {
    if (!pImpl->evaluator) {
        static std::unordered_map<SentimentLabel,
               std::unordered_map<SentimentLabel, int>> emptyMatrix;
        return emptyMatrix;
    }

    return pImpl->evaluator->getConfusionMatrix();
}

} // namespace sentiment
