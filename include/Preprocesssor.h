#ifndef PREPROCESSOR_H
#define PREPROCESSOR_H

#include <string>
#include <vector>
#include <unordered_set>

namespace sentiment {

/**
 * @brief Class for text preprocessing and tokenization
 *
 * This class handles text cleaning, tokenization, and optional
 * stop word removal for NLP preprocessing.
 */
class Preprocessor {
public:
    /**
     * @brief Constructor with optional stop words
     * @param useStopWords Whether to remove stop words
     */
    explicit Preprocessor(bool useStopWords = true);

    /**
     * @brief Clean and normalize text
     *
     * This function:
     * - Converts text to lowercase
     * - Removes punctuation
     * - Normalizes whitespace
     *
     * @param text Input text to clean
     * @return Cleaned text
     */
    std::string cleanText(const std::string& text) const;

    /**
     * @brief Tokenize text into words
     * @param text Text to tokenize
     * @return Vector of tokens (words)
     */
    std::vector<std::string> tokenize(const std::string& text) const;

    /**
     * @brief Clean and tokenize text in one step
     * @param text Text to preprocess
     * @return Vector of tokens
     */
    std::vector<std::string> preprocess(const std::string& text) const;

    /**
     * @brief Add custom stop words
     * @param words Vector of words to add as stop words
     */
    void addStopWords(const std::vector<std::string>& words);

    /**
     * @brief Check if a word is a stop word
     * @param word Word to check
     * @return true if the word is a stop word, false otherwise
     */
    bool isStopWord(const std::string& word) const;

private:
    bool useStopWords; ///< Whether to use stop word removal
    std::unordered_set<std::string> stopWords; ///< Set of stop words

    /**
     * @brief Initialize default English stop words
     */
    void initializeStopWords();
};

} // namespace sentiment

#endif // PREPROCESSOR_H
