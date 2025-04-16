#include <gtest/gtest.h>
#include <string>
#include <vector>
#include "preprocessor.h"

using namespace sentiment;

// Test fixture for Preprocessor tests
class PreprocessorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create preprocessor instances for testing
        preprocessorWithStopWords = new Preprocessor(true);
        preprocessorWithoutStopWords = new Preprocessor(false);
    }

    void TearDown() override {
        delete preprocessorWithStopWords;
        delete preprocessorWithoutStopWords;
    }

    Preprocessor* preprocessorWithStopWords;
    Preprocessor* preprocessorWithoutStopWords;
};

// Test text cleaning functionality
TEST_F(PreprocessorTest, CleanText) {
    // Test lowercasing
    EXPECT_EQ(preprocessorWithStopWords->cleanText("Hello World"), "hello world");

    // Test punctuation removal
    EXPECT_EQ(preprocessorWithStopWords->cleanText("Hello, World!"), "hello world");

    // Test multiple spaces
    EXPECT_EQ(preprocessorWithStopWords->cleanText("Hello   World"), "hello world");

    // Test mixed case and punctuation
    EXPECT_EQ(preprocessorWithStopWords->cleanText("Hello, WORLD!!!"), "hello world");
}

// Test tokenization without stop word removal
TEST_F(PreprocessorTest, TokenizeWithoutStopWords) {
    std::string text = "this is a test";
    std::vector<std::string> expected = {"this", "is", "a", "test"};

    std::vector<std::string> tokens = preprocessorWithoutStopWords->tokenize(text);

    ASSERT_EQ(tokens.size(), expected.size());
    for (size_t i = 0; i < tokens.size(); ++i) {
        EXPECT_EQ(tokens[i], expected[i]);
    }
}

// Test tokenization with stop word removal
TEST_F(PreprocessorTest, TokenizeWithStopWords) {
    std::string text = "this is a test with some stop words";
    std::vector<std::string> tokens = preprocessorWithStopWords->tokenize(text);

    // The "this", "is", "a", "with", "some" should be removed as stop words
    EXPECT_NE(std::find(tokens.begin(), tokens.end(), "test"), tokens.end());
    EXPECT_NE(std::find(tokens.begin(), tokens.end(), "words"), tokens.end());

    // These should be removed as stop words
    EXPECT_EQ(std::find(tokens.begin(), tokens.end(), "this"), tokens.end());
    EXPECT_EQ(std::find(tokens.begin(), tokens.end(), "is"), tokens.end());
    EXPECT_EQ(std::find(tokens.begin(), tokens.end(), "a"), tokens.end());
    EXPECT_EQ(std::find(tokens.begin(), tokens.end(), "with"), tokens.end());
    EXPECT_EQ(std::find(tokens.begin(), tokens.end(), "some"), tokens.end());
}

// Test full preprocessing pipeline
TEST_F(PreprocessorTest, Preprocess) {
    std::string text = "This, is a TEST with PUNCTUATION!!!";

    std::vector<std::string> tokens = preprocessorWithoutStopWords->preprocess(text);

    // Check that text was lowercased and punctuation removed
    ASSERT_FALSE(tokens.empty());
    EXPECT_EQ(std::find(tokens.begin(), tokens.end(), "TEST"), tokens.end());
    EXPECT_NE(std::find(tokens.begin(), tokens.end(), "test"), tokens.end());

    // Check that "PUNCTUATION" was properly processed
    EXPECT_EQ(std::find(tokens.begin(), tokens.end(), "PUNCTUATION"), tokens.end());
    EXPECT_NE(std::find(tokens.begin(), tokens.end(), "punctuation"), tokens.end());
}

// Test main function (required for Google Test)
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
