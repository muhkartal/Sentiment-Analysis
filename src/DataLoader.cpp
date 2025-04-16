#include "DataLoader.h"
#include <fstream>
#include <sstream>
#include <stdexcept> // For std::runtime_error
#include <iostream>  // For error messages

namespace sentiment
{

std::vector<Document> DataLoader::loadData(const std::string& filepath)
{
    std::vector<Document> data;
    std::ifstream file(filepath);

    if (!file.is_open()) {
        throw std::runtime_error("DataLoader Error: Could not open file: " + filepath);
    }

    std::string line;
    int line_number = 0;
    while (std::getline(file, line)) {
        line_number++;
        std::stringstream ss(line);
        std::string segment;
        std::vector<std::string> seglist;

        // Split the line by comma
        while (std::getline(ss, segment, ',')) {
            seglist.push_back(segment);
        }

        // Expecting exactly two segments: text and label
        if (seglist.size() >= 2) {
            // Handle cases where text might contain commas by joining initial segments
            std::string text = seglist[0];
            for(size_t i = 1; i < seglist.size() - 1; ++i) {
                text += "," + seglist[i]; // Re-add comma removed by splitting
            }
            std::string label = seglist.back(); // Last segment is the label

            // Basic trimming of whitespace from label
            label.erase(0, label.find_first_not_of(" \t\n\r\f\v"));
            label.erase(label.find_last_not_of(" \t\n\r\f\v") + 1);

            data.push_back({text, label});
        } else {
            // Optionally warn about malformed lines
             std::cerr << "DataLoader Warning: Skipping malformed line "
                       << line_number << " in file " << filepath << std::endl;
        }
    }

    file.close();
    return data;
}

} // namespace sentiment
