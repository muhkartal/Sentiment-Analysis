#include "Preprocessor.h"
#include <algorithm> // For std::transform, std::remove_if
#include <cctype>    // For ::tolower, ::isspace, ::ispunct
#include <sstream>   // For std::stringstream
#include <fstream>   // For loading stop words
#include <iterator>  // For std::istream_iterator

namespace sentiment
{

Preprocessor::Preprocessor() : remove_stop_words_(false) {
    // Initialize with a small, common English stop word list if desired
    // Or rely solely on loading from file via loadStopWords()
     stop_words_ = {"a", "an", "the", "in", "on", "at", "to", "for", "is",
                    "am", "are", "was", "were", "i", "you", "he", "she",
                    "it", "we", "they", "and", "or", "but", "so", "this",
                    "that", "these", "those", "my", "your", "his", "her",
                    "its", "our", "their"};
}

std::string Preprocessor::toLower(const std::string& text) const {
    std::string lower_text = text;
    std::transform(lower_text.begin(), lower_text.end(), lower_text.begin(),
                   [](unsigned char c){ return std::tolower(c); });
    return lower_text;
}

std::string Preprocessor::removePunctuation(const std::string& text) const {
    std::string no_punct_text;
    std::copy_if(text.begin(), text.end(), std::back_inserter(no_punct_text),
                 [](unsigned char c){ return !std::ispunct(c); });
    return no_punct_text;
}

std::vector<std::string> Preprocessor::tokenize(const std::string& text) const {
    std::stringstream ss(text);
    std::string word;
    std::vector<std::string> tokens;
    while (ss >> word) { // Splits by whitespace
        tokens.push_back(word);
    }
    return tokens;
}

void Preprocessor::setStopWordRemoval(bool remove) {
    remove_stop_words_ = remove;
}

bool Preprocessor::loadStopWords(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        return false; // Indicate failure
    }
    stop_words_.clear(); // Clear existing stop words
    std::string word;
    while (file >> word) {
        stop_words_.insert(toLower(word)); // Store stop words in lowercase
    }
    file.close();
    return true;
}


std::vector<std::string> Preprocessor::process(const std::string& text) const {
    // 1. Convert to lowercase
    std::string current_text = toLower(text);

    // 2. Remove punctuation
    current_text = removePunctuation(current_text);

    // 3. Tokenize
    std::vector<std::string> tokens = tokenize(current_text);

    // 4. (Optional) Remove stop words
    if (remove_stop_words_) {
        std::vector<std::string> filtered_tokens;
        for (const auto& token : tokens) {
            if (stop_words_.find(token) == stop_words_.end()) {
                // Keep token if it's NOT in the stop word list
                filtered_tokens.push_back(token);
            }
        }
        return filtered_tokens; // Return the filtered list
    }

    return tokens; // Return tokens without stop word removal
}


} // namespace sentiment
