#ifndef MODEL_H
#define MODEL_H

#include <vector>
#include "utils.h"

namespace sentiment {

/**
 * @brief Abstract base class for sentiment classification models
 *
 * This class defines the interface that all classification models must implement.
 */
class Model {
public:
    /**
     * @brief Virtual destructor
     */
    virtual ~Model() = default;

    /**
     * @brief Train the model on feature vectors
     * @param trainingData Vector of FeatureVector structures containing features and labels
     * @return true if training was successful, false otherwise
     */
    virtual bool train(const std::vector<FeatureVector>& trainingData) = 0;

    /**
     * @brief Predict sentiment label for a feature vector
     * @param features Input feature vector
     * @return Predicted SentimentLabel
     */
    virtual SentimentLabel predict(const std::vector<double>& features) const = 0;

    /**
     * @brief Check if the model is trained
     * @return true if the model is trained, false otherwise
     */
    virtual bool isTrained() const = 0;

    /**
     * @brief Get model name
     * @return String containing model name
     */
    virtual std::string getName() const = 0;
};

} // namespace sentiment

#endif // MODEL_H
