#ifndef DATALOADER_H
#define DATALOADER_H

#include <string>
#include <vector>
#include <utility> // For std::pair

namespace sentiment
{

/**
 * @brief Represents a single data instance (text and its label).
 */
struct Document {
    std::string text;
    std::string label;
};

/**
 * @brief Loads sentiment analysis data from a file.
 *
 * Assumes a CSV-like format where each line contains "text,label".
 * Does not handle escaped commas within the text itself for simplicity.
 */
class DataLoader
{
public:
    /**
     * @brief Loads data from the specified file path.
     * @param filepath Path to the data file.
     * @return A vector of Document objects.
     * @throws std::runtime_error if the file cannot be opened.
     */
    static std::vector<Document> loadData(const std::string& filepath);
};

} // namespace sentiment

#endif // DATALOADER_H
