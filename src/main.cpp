#include <iostream>
#include <vector>
#include <string>
#include <random>      // For std::shuffle, std::mt19937, std::random_device
#include <algorithm>   // For std::shuffle
#include <numeric>     // For std::iota
#include <iomanip>     // For std::setprecision

#include "DataLoader.h"
#include "Preprocessor.h"
#include "FeatureExtractor.h"
#include "NaiveBayesClassifier.h"

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_data_csv>" << std::endl;
        return 1;
    }

    std::string data_filepath = argv[1];

    try {
        // --- 1. Load Data ---
        std::cout << "Loading data from: " << data_filepath << "..." << std::endl;
        auto documents = sentiment::DataLoader::loadData(data_filepath);
        if (documents.empty()) {
            std::cerr << "Error: No documents loaded. Check data file format and path." << std::endl;
            return 1;
        }
        std::cout << "Loaded " << documents.size() << " documents." << std::endl;

        // --- 2. Preprocessing ---
        std::cout << "Preprocessing data..." << std::endl;
        sentiment::Preprocessor preprocessor;
        // Optional: Enable stop word removal
        // preprocessor.setStopWordRemoval(true);
        // Optional: Load stop words from a file
        // if (!preprocessor.loadStopWords("path/to/stopwords.txt")) {
        //     std::cerr << "Warning: Could not load stop words file." << std::endl;
        // }

        std::vector<std::vector<std::string>> all_tokenized_docs;
        std::vector<std::string> all_labels;
        all_tokenized_docs.reserve(documents.size());
        all_labels.reserve(documents.size());

        for (const auto& doc : documents) {
            all_tokenized_docs.push_back(preprocessor.process(doc.text));
            all_labels.push_back(doc.label);
        }
        std::cout << "Preprocessing complete." << std::endl;

        // --- 3. Data Splitting (Train/Validation) ---
        if (documents.size() < 5) {
             std::cerr << "Warning: Very small dataset (" << documents.size()
                       << " docs). Evaluation might not be meaningful. Consider more data." << std::endl;
             // Handle very small datasets - maybe use all for training or skip validation?
             // For this example, we'll proceed but results will be unreliable.
             if (documents.empty()){ return 1; } // Exit if truly empty after warning
        }

        double train_split_ratio = 0.8;
        size_t total_samples = documents.size();
        size_t train_size = static_cast<size_t>(total_samples * train_split_ratio);
        size_t validation_size = total_samples - train_size;

        if (train_size == 0 || validation_size == 0) {
             std::cerr << "Error: Dataset too small to create both training and validation sets with ratio "
                       << train_split_ratio << ". Need at least " << static_cast<int>(1.0/std::min(train_split_ratio, 1.0 - train_split_ratio)) + 1
                       << " samples." << std::endl;
             // Decide how to proceed: maybe use all data for training and skip validation?
             // For now, let's just make validation size at least 1 if possible.
             if (total_samples > 1) {
                 train_size = total_samples - 1;
                 validation_size = 1;
             } else {
                  train_size = total_samples; // Use all for training if only 1 sample
                  validation_size = 0;
                  std::cout << "Warning: Only 1 sample, using it for training. No validation possible." << std::endl;
             }
        }


        std::vector<size_t> indices(total_samples);
        std::iota(indices.begin(), indices.end(), 0); // Fill with 0, 1, 2, ...

        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(indices.begin(), indices.end(), g); // Shuffle indices randomly

        std::vector<std::vector<std::string>> train_tokenized_docs;
        std::vector<std::string> train_labels;
        std::vector<std::vector<std::string>> validation_tokenized_docs;
        std::vector<std::string> validation_labels;

        train_tokenized_docs.reserve(train_size);
        train_labels.reserve(train_size);
        validation_tokenized_docs.reserve(validation_size);
        validation_labels.reserve(validation_size);

        for (size_t i = 0; i < train_size; ++i) {
            train_tokenized_docs.push_back(all_tokenized_docs[indices[i]]);
            train_labels.push_back(all_labels[indices[i]]);
        }
        for (size_t i = train_size; i < total_samples; ++i) {
            validation_tokenized_docs.push_back(all_tokenized_docs[indices[i]]);
            validation_labels.push_back(all_labels[indices[i]]);
        }

        std::cout << "Split data: " << train_size << " training, " << validation_size << " validation samples." << std::endl;


        // --- 4. Feature Extraction ---
        std::cout << "Building vocabulary and extracting features..." << std::endl;
        sentiment::FeatureExtractor feature_extractor;
        feature_extractor.buildVocabulary(train_tokenized_docs); // Build vocab ONLY on training data
        size_t vocab_size = feature_extractor.getVocabularySize();
         std::cout << "Vocabulary size: " << vocab_size << std::endl;
         if (vocab_size == 0) {
            std::cerr << "Error: Vocabulary size is 0. Check training data or preprocessing." << std::endl;
            return 1;
         }


        std::vector<sentiment::FeatureExtractor::FeatureVector> train_features;
        train_features.reserve(train_size);
        for (const auto& doc_tokens : train_tokenized_docs) {
            train_features.push_back(feature_extractor.extractFeatures(doc_tokens));
        }

        std::vector<sentiment::FeatureExtractor::FeatureVector> validation_features;
         if (validation_size > 0) {
            validation_features.reserve(validation_size);
            for (const auto& doc_tokens : validation_tokenized_docs) {
                validation_features.push_back(feature_extractor.extractFeatures(doc_tokens));
            }
         }
        std::cout << "Feature extraction complete." << std::endl;

        // --- 5. Model Training ---
        std::cout << "Training Naive Bayes classifier..." << std::endl;
        sentiment::NaiveBayesClassifier classifier(1.0); // Using Laplace smoothing alpha=1.0
        classifier.train(train_features, train_labels, vocab_size);
        std::cout << "Training complete. Model trained on classes: ";
        const auto& trained_classes = classifier.getClasses();
        for(size_t i = 0; i < trained_classes.size(); ++i) {
            std::cout << trained_classes[i] << (i == trained_classes.size() - 1 ? "" : ", ");
        }
        std::cout << std::endl;


        // --- 6. Evaluation ---
        if (validation_size > 0) {
            std::cout << "Evaluating model on validation set..." << std::endl;
            int correct_predictions = 0;
            for (size_t i = 0; i < validation_size; ++i) {
                std::string predicted_label = classifier.predict(validation_features[i]);
                if (predicted_label == validation_labels[i]) {
                    correct_predictions++;
                }
                // Optional: Print individual predictions for debugging
                // std::cout << "  Doc " << i << ": Actual='" << validation_labels[i]
                //           << "', Predicted='" << predicted_label << "'" << std::endl;
            }

            double accuracy = static_cast<double>(correct_predictions) / validation_size;
            std::cout << "----------------------------------------" << std::endl;
            std::cout << "Evaluation Results (Validation Set):" << std::endl;
            std::cout << "  Correct Predictions: " << correct_predictions << " / " << validation_size << std::endl;
            std::cout << "  Accuracy: " << std::fixed << std::setprecision(4) << accuracy * 100.0 << "%" << std::endl;
            std::cout << "----------------------------------------" << std::endl;
        } else {
             std::cout << "Skipping evaluation as validation set size is 0." << std::endl;
        }


        // --- 7. Command-Line Inference ---
        std::cout << "\n--- Interactive Sentiment Analysis ---" << std::endl;
        std::string input_text;
        while (true) {
            std::cout << "Enter text to classify (or press Enter to quit): ";
            std::getline(std::cin, input_text);

            if (input_text.empty()) {
                break;
            }

            // Preprocess the input text
            auto tokens = preprocessor.process(input_text);

            // Extract features using the *trained* vocabulary
            auto features = feature_extractor.extractFeatures(tokens);

            // Predict using the *trained* classifier
            std::string prediction = classifier.predict(features);

            std::cout << "  -> Predicted Sentiment: " << prediction << std::endl;
        }

        std::cout << "Exiting." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
