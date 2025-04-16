#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <iomanip>
#include <fstream>

#include "data_loader.h"
#include "preprocessor.h"
#include "feature_extractor.h"
#include "naive_bayes.h"
#include "evaluator.h"
#include "utils.h"

using namespace sentiment;

// Function to print program header
void printHeader() {
    std::cout << "====================================================\n";
    std::cout << "          C++ Sentiment Analysis Pipeline           \n";
    std::cout << "====================================================\n";
}

// Function to print usage information
void printUsage(const std::string& programName) {
    std::cout << "Usage: " << programName << " [options]\n";
    std::cout << "Options:\n";
    std::cout << "  --file FILE      Path to training data CSV file\n";
    std::cout << "  --interactive    Enable interactive mode for inference\n";
    std::cout << "  --help           Display this help message\n";
}

// Function to parse command line arguments
std::unordered_map<std::string, std::string> parseArgs(int argc, char* argv[]) {
    std::unordered_map<std::string, std::string> args;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help") {
            args["help"] = "true";
        } else if (arg == "--interactive") {
            args["interactive"] = "true";
        } else if (arg == "--file" && i + 1 < argc) {
            args["file"] = argv[++i];
        } else if (arg.substr(0, 2) == "--") {
            std::cerr << "Unknown option: " << arg << std::endl;
        }
    }

    return args;
}

// Function for interactive mode
void runInteractiveMode(
    const Preprocessor& preprocessor,
    const FeatureExtractor& featureExtractor,
    const Model& model
) {
    std::cout << "\n--- Interactive Mode ---\n";
    std::cout << "Enter text to analyze sentiment (type 'exit' to quit):\n";

    std::string input;
    while (true) {
        std::cout << "\n> ";
        std::getline(std::cin, input);

        // Check for exit command
        if (input == "exit" || input == "quit") {
            break;
        }

        // Skip empty input
        if (input.empty()) {
            continue;
        }

        // Extract features and predict
        std::vector<double> features = featureExtractor.extractFeatures(input);
        SentimentLabel prediction = model.predict(features);

        // Print prediction
        std::cout << "Sentiment: " << sentimentToString(prediction) << std::endl;
    }
}

// Function to create sample data file if not provided
std::string createSampleDataFile() {
    std::string filePath = "data/sample_data.csv";
    std::ofstream file(filePath);

    if (file.is_open()) {
        file << "text,sentiment\n";
        file << "\"I love this product, it's amazing!\",positive\n";
        file << "\"This is the worst experience ever.\",negative\n";
        file << "\"The service was okay, nothing special.\",neutral\n";
        file << "\"I'm extremely happy with my purchase.\",positive\n";
        file << "\"The quality was disappointing, I expected better.\",negative\n";
        file << "\"It works as expected, no problems so far.\",neutral\n";
        file << "\"I absolutely hate how this performs.\",negative\n";
        file << "\"Best decision I ever made, highly recommend!\",positive\n";
        file << "\"The price is reasonable for what you get.\",neutral\n";
        file << "\"Complete waste of money, avoid at all costs.\",negative\n";
        file << "\"I'm satisfied with this product.\",positive\n";
        file << "\"Not impressed but not terrible either.\",neutral\n";
        file << "\"The customer service was excellent.\",positive\n";
        file << "\"I regret buying this, total garbage.\",negative\n";
        file << "\"It's fine, does the job adequately.\",neutral\n";
        file << "\"I can't believe how good this is!\",positive\n";
        file << "\"Very disappointed with the result.\",negative\n";
        file << "\"Average performance, nothing to write home about.\",neutral\n";
        file << "\"This exceeded all my expectations!\",positive\n";
        file << "\"Terrible design, confusing interface.\",negative\n";
        file.close();

        std::cout << "Created sample data file: " << filePath << std::endl;
    } else {
        std::cerr << "Error: Could not create sample data file" << std::endl;
        return "";
    }

    return filePath;
}

int main(int argc, char* argv[]) {
    printHeader();

    // Parse command line arguments
    auto args = parseArgs(argc, argv);

    // Check for help flag
    if (args.count("help") > 0) {
        printUsage(argv[0]);
        return 0;
    }

    // Timer for measuring performance
    auto startTime = std::chrono::high_resolution_clock::now();

    // Initialize file path
    std::string filePath = args.count("file") > 0 ? args["file"] : "data/sample_data.csv";

    // Create sample data if file doesn't exist
    std::ifstream fileCheck(filePath);
    if (!fileCheck.good()) {
        std::cout << "File not found: " << filePath << std::endl;
        filePath = createSampleDataFile();
        if (filePath.empty()) {
            return 1;
        }
    }

    // 1. Data Loading
    std::cout << "\n--- Step 1: Loading Data ---\n";
    DataLoader dataLoader;
    bool loadSuccess = dataLoader.loadFromCSV(filePath);
    if (!loadSuccess) {
        std::cerr << "Error: Failed to load data from " << filePath << std::endl;
        return 1;
    }

    std::cout << "Loaded " << dataLoader.getData().size() << " examples from "
              << filePath << std::endl;

    // Split data into training and validation sets
    auto [trainData, validData] = dataLoader.splitTrainValidation(0.8);
    std::cout << "Split data into " << trainData.size() << " training examples and "
              << validData.size() << " validation examples" << std::endl;

    // 2. Preprocessing and Feature Extraction
    std::cout << "\n--- Step 2: Preprocessing and Feature Extraction ---\n";
    Preprocessor preprocessor(true); // Use stop word removal
    FeatureExtractor featureExtractor(
        preprocessor,
        FeatureExtractor::Method::BAG_OF_WORDS
    );

    // Build vocabulary from training data
    featureExtractor.buildVocabulary(trainData, 2, 5000);

    // Transform training and validation data to feature vectors
    std::vector<FeatureVector> trainFeatures = featureExtractor.batchTransform(trainData);
    std::vector<FeatureVector> validFeatures = featureExtractor.batchTransform(validData);

    std::cout << "Extracted features with vocabulary size: "
              << featureExtractor.getVocabularySize() << std::endl;

    // 3. Model Training
    std::cout << "\n--- Step 3: Model Training ---\n";
    NaiveBayes model(1.0); // Alpha = 1.0 (Laplace smoothing)
    bool trainSuccess = model.train(trainFeatures);
    if (!trainSuccess) {
        std::cerr << "Error: Failed to train model" << std::endl;
        return 1;
    }

    // 4. Evaluation
    std::cout << "\n--- Step 4: Evaluation ---\n";
    Evaluator evaluator(model);
    EvaluationMetrics metrics = evaluator.evaluate(validFeatures);
    evaluator.printResults();

    // Print execution time
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        endTime - startTime).count();
    std::cout << "Total execution time: " << duration / 1000.0 << " seconds\n";

    // 5. Interactive mode (if requested)
    if (args.count("interactive") > 0) {
        runInteractiveMode(preprocessor, featureExtractor, model);
    } else {
        std::cout << "\nRun with --interactive flag to test the model with custom input\n";
    }

    return 0;
}
