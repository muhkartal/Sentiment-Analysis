#ifndef PREPROCESSOR_H
#define PREPROCESSOR_H

#include <string>
#include <vector>
#include <unordered_set>

namespace sentiment
{

/**
 * @brief Performs basic text preprocessing steps.
 */
class Preprocessor
{
public:
    /**
     * @brief Default constructor. Initializes default settings.
     */
    Preprocessor();

    /**
     * @brief Processes a single text string.
     *
     * Applies lowercase conversion, punctuation removal, and tokenization.
     * Optionally removes stop words if enabled.
     *
     * @param text The input text string.
     * @return A vector of processed tokens.
     */
    std::vector<std::string> process(const std::string& text) const;

    /**
     * @brief Enables or disables stop word removal.
     * @param remove True to enable, false to disable.
     */
    void setStopWordRemoval(bool remove);

    /**
     * @brief Loads stop words from a file (one word per line).
     * @param filepath Path to the stop words file.
     * @return True if loading was successful, false otherwise.
     */
    bool loadStopWords(const std::string& filepath);


private:
    /**
     * @brief Converts a string to lowercase.
     * @param text Input string.
     * @return Lowercase string.
     */
    std::string toLower(const std::string& text) const;

    /**
     * @brief Removes punctuation characters from a string.
     * @param text Input string.
     * @return String with punctuation removed.
     */
    std::string removePunctuation(const std::string& text) const;

    /**
     * @brief Splits a string into tokens based on whitespace.
     * @param text Input string.
     * @return Vector of tokens.
     */
    std::vector<std::string> tokenize(const std::string& text) const;

    bool remove_stop_words_ = false;
    std::unordered_set<std::string> stop_words_;
};

} // namespace sentiment

#endif // PREPROCESSOR_H
