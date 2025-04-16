#include "utils.h"
#include <algorithm>
#include <stdexcept>

namespace sentiment {

SentimentLabel stringToSentiment(const std::string& sentiment) {
    // Convert sentiment string to lowercase for case-insensitive comparison
    std::string sentimentLower = sentiment;
    std::transform(sentimentLower.begin(), sentimentLower.end(),
                   sentimentLower.begin(), ::tolower);

    if (sentimentLower == "positive" || sentimentLower == "pos") {
        return SentimentLabel::POSITIVE;
    } else if (sentimentLower == "negative" || sentimentLower == "neg") {
        return SentimentLabel::NEGATIVE;
    } else if (sentimentLower == "neutral" || sentimentLower == "neu") {
        return SentimentLabel::NEUTRAL;
    } else {
        return SentimentLabel::UNKNOWN;
    }
}

std::string sentimentToString(SentimentLabel label) {
    switch (label) {
        case SentimentLabel::POSITIVE:
            return "positive";
        case SentimentLabel::NEGATIVE:
            return "negative";
        case SentimentLabel::NEUTRAL:
            return "neutral";
        case SentimentLabel::UNKNOWN:
        default:
            return "unknown";
    }
}

} // namespace sentiment
