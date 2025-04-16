#include "data_loader.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdexcept>

namespace sentiment {

bool DataLoader::loadFromCSV(
    const std::string& filePath,
    bool hasHeader,
    int textColumn,
    int labelColumn
) {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filePath << std::endl;
        return false;
    }

    data.clear();
    std::string line;

    // Skip header if present
    if (hasHeader && std::getline(file, line)) {
        // Header line is ignored
    }

    // Process data rows
    while (std::getline(file, line)) {
        std::istringstream ss(line);
        std::string token;
        std::vector<std::string> tokens;

        // Parse CSV line
        while (std::getline(ss, token, ',')) {
            // Handle quoted fields
            if (!token.empty() && token.front() == '"') {
                // Remove starting quote
                token = token.substr(1);

                // If the field doesn't end with a quote, read more tokens
                if (token.empty() || token.back() != '"') {
                    std::string more;
                    while (std::getline(ss, more, ',')) {
                        token += "," + more;
                        if (!more.empty() && more.back() == '"') {
                            break;
                        }
                    }
                }

                // Remove ending quote if present
                if (!token.empty() && token.back() == '"') {
                    token.pop_back();
                }
            }

            tokens.push_back(token);
        }

        // Ensure we have enough columns
        if (static_cast<size_t>(std::max(textColumn, labelColumn) + 1) > tokens.size()) {
            std::cerr << "Warning: Line doesn't have enough columns: " << line << std::endl;
            continue;
        }

        TextData textData;
        textData.text = tokens[textColumn];
        textData.label = stringToSentiment(tokens[labelColumn]);

        // Only add data with valid labels
        if (textData.label != SentimentLabel::UNKNOWN) {
            data.push_back(textData);
        }
    }

    file.close();

    if (data.empty()) {
        std::cerr << "Warning: No valid data loaded from file" << std::endl;
        return false;
    }

    return true;
}

const std::vector<TextData>& DataLoader::getData() const {
    return data;
}

std::pair<std::vector<TextData>, std::vector<TextData>>
DataLoader::splitTrainValidation(double trainRatio) const {
    return trainValidationSplit(data, trainRatio);
}

} // namespace sentiment
