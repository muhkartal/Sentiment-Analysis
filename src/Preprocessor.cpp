#include "preprocessor.h"
#include <algorithm>
#include <cctype>
#include <sstream>
#include <regex>

namespace sentiment {

Preprocessor::Preprocessor(bool useStopWords) : useStopWords(useStopWords) {
    if (useStopWords) {
        initializeStopWords();
    }
}

std::string Preprocessor::cleanText(const std::string& text) const {
    // Convert to lowercase
    std::string cleanedText = text;
    std::transform(cleanedText.begin(), cleanedText.end(),
                  cleanedText.begin(), ::tolower);

    // Replace punctuation with spaces
    std::regex punctRegex("[\\!\\\"\\#\\$\\%\\&\\'\\(\\)\\*\\+\\,\\-\\.\\/\\:\\;\\<\\=\\>\\?\\@\\[\\\\\\]\\^_\\`\\{\\|\\}\\~]");
    cleanedText = std::regex_replace(cleanedText, punctRegex, " ");

    // Replace multiple spaces with a single space
    std::regex multipleSpacesRegex("\\s+");
    cleanedText = std::regex_replace(cleanedText, multipleSpacesRegex, " ");

    // Trim leading and trailing spaces
    cleanedText = std::regex_replace(cleanedText, std::regex("^\\s+|\\s+$"), "");

    return cleanedText;
}

std::vector<std::string> Preprocessor::tokenize(const std::string& text) const {
    std::vector<std::string> tokens;
    std::istringstream iss(text);
    std::string token;

    while (iss >> token) {
        // Skip stop words if enabled
        if (useStopWords && isStopWord(token)) {
            continue;
        }

        tokens.push_back(token);
    }

    return tokens;
}

std::vector<std::string> Preprocessor::preprocess(const std::string& text) const {
    return tokenize(cleanText(text));
}

void Preprocessor::addStopWords(const std::vector<std::string>& words) {
    for (const auto& word : words) {
        stopWords.insert(word);
    }
}

bool Preprocessor::isStopWord(const std::string& word) const {
    return stopWords.find(word) != stopWords.end();
}

void Preprocessor::initializeStopWords() {
    // Common English stop words
    const std::vector<std::string> defaultStopWords = {
        "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
        "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
        "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
        "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
        "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
        "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
        "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
        "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
        "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
        "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
        "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
        "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than",
        "that", "that's", "the", "their", "theirs", "them", "themselves", "then",
        "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've",
        "this", "those", "through", "to", "too", "under", "until", "up", "very", "was",
        "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what",
        "what's", "when", "when's", "where", "where's", "which", "while", "who",
        "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you",
        "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"
    };

    stopWords.insert(defaultStopWords.begin(), defaultStopWords.end());
}

} // namespace sentiment
