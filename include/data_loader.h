#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include <string>
#include <vector>
#include "utils.h"

namespace sentiment {

/**
 * @brief Class for loading text data from files
 *
 * This class handles loading text data from CSV or text files,
 * parsing the sentiment labels, and provides functionality for
 * splitting into training and validation sets.
 */
class DataLoader {
public:
    /**
     * @brief Default constructor
     */
    DataLoader() = default;

    /**
     * @brief Load data from a CSV file
     * @param filePath Path to the CSV file
     * @param hasHeader Whether the CSV file has a header row
     * @param textColumn Index of the column containing text data
     * @param labelColumn Index of the column containing sentiment labels
     * @return true if loading was successful, false otherwise
     */
    bool loadFromCSV(
        const std::string& filePath,
        bool hasHeader = true,
        int textColumn = 0,
        int labelColumn = 1
    );

    /**
     * @brief Get all loaded text data
     * @return Vector of TextData structures
     */
    const std::vector<TextData>& getData() const;

    /**
     * @brief Split data into training and validation sets
     * @param trainRatio Ratio of data to use for training (between 0 and 1)
     * @return A pair of vectors (train_data, validation_data)
     */
    std::pair<std::vector<TextData>, std::vector<TextData>>
    splitTrainValidation(double trainRatio = 0.8) const;

private:
    std::vector<TextData> data; ///< Loaded text data with labels
};

} // namespace sentiment

#endif // DATA_LOADER_H
