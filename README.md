# C++ Sentiment Analysis Engine

<div align="center">

[![C++ Standard](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://isocpp.org/std/the-standard)
[![Build System](https://img.shields.io/badge/Build-CMake-blue)](https://cmake.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-README-brightgreen.svg)](#documentation)
**A foundational C++ application for text sentiment classification using Naive Bayes.**

[Overview](#overview) • [Features](#key-features) • [Installation](#installation) • [Quick Start](#quick-start) • [Documentation](#documentation) • [Contributing](#contributing)

</div>

## Overview

This project implements a sentiment analysis system (Positive, Negative, Neutral) from the ground up in C++17. It serves as a practical example of applying fundamental Natural Language Processing (NLP) and Machine Learning (ML) concepts within a well-structured, performant C++ environment. Built with CMake for cross-platform compatibility, it demonstrates core algorithm implementation (Naive Bayes, Bag-of-Words) and good software engineering practices in C++.

## Key Features

-  **Standard C++17 Implementation**: Core logic uses standard library features, minimizing external dependencies.
-  **Modular Architecture**: Code organized into logical components (`DataLoader`, `Preprocessor`, `FeatureExtractor`, `NaiveBayesClassifier`).
-  **Naive Bayes Classifier**: Implemented from scratch with Laplace smoothing.
-  **Bag-of-Words (BoW)**: Simple and effective text feature extraction.
-  **Text Preprocessing**: Includes lowercasing, punctuation removal, and tokenization.
-  **CMake Build System**: Ensures robust, cross-platform building (Linux, macOS, Windows).
-  **Train/Validation Split**: Automatic data partitioning for model evaluation.
-  **Accuracy Evaluation**: Reports model performance on validation data.
-  **Interactive Inference**: Classify new text inputs via the command line after training.
-  **Clean Code & Documentation**: Focus on readability, Doxygen-style comments, and clear structure.

## Components

| Component              | Description                                                                     | Header File                      | Source File                    |
| ---------------------- | ------------------------------------------------------------------------------- | -------------------------------- | ------------------------------ |
| `DataLoader`           | Loads and parses text data and labels from a CSV file.                          | `include/DataLoader.h`           | `src/DataLoader.cpp`           |
| `Preprocessor`         | Cleans text (lowercase, punctuation) and tokenizes it into words.               | `include/Preprocessor.h`         | `src/Preprocessor.cpp`         |
| `FeatureExtractor`     | Builds vocabulary from training data and creates Bag-of-Words feature vectors.  | `include/FeatureExtractor.h`     | `src/FeatureExtractor.cpp`     |
| `NaiveBayesClassifier` | Implements training and prediction logic for the Naive Bayes algorithm.         | `include/NaiveBayesClassifier.h` | `src/NaiveBayesClassifier.cpp` |
| `main`                 | Orchestrates the pipeline, handles arguments, evaluation, and interactive mode. | N/A                              | `src/main.cpp`                 |

## Real-World Use Cases

-  **Social Media Monitoring**: Track public opinion and brand sentiment.
-  **Customer Feedback Analysis**: Automatically classify reviews, surveys, and support tickets.
-  **Market Research**: Gauge audience reaction to products or campaigns.
-  **Content Recommendation**: Suggest articles or media based on user sentiment.
-  **Chatbot Enhancement**: Understand user emotion for better interaction.

## Installation

This section details how to build the project using CMake.

### Prerequisites

-  **C++ Compiler:** Supporting C++17 (e.g., GCC 7+, Clang 5+, MSVC 19.14+ / VS 2017+).
-  **CMake:** Version 3.14 or higher ([Download CMake](https://cmake.org/download/)).
-  **Git:** For cloning the repository ([Download Git](https://git-scm.com/downloads)).

### Build Steps

1. **Clone the Repository:**

   ```bash
   git clone <your-repository-url> sentiment-analysis-cpp
   cd sentiment-analysis-cpp
   ```

   _(Replace `<your-repository-url>` with the actual URL)_

2. **Create Build Directory:**

   ```bash
   mkdir build
   cd build
   ```

3. **Configure with CMake:**

   -  **Linux/macOS/MinGW:**
      ```bash
      # Debug build (default)
      cmake ..
      # Release build (optimized)
      # cmake .. -DCMAKE_BUILD_TYPE=Release
      ```
   -  **Windows (Visual Studio):**
      ```bash
      # Example for VS 2022 64-bit
      cmake .. -G "Visual Studio 17 2022" -A x64
      ```

4. **Compile the Project:**
   -  **CMake Command (Recommended):**
      ```bash
      # Build default target (uses CMAKE_BUILD_TYPE if set)
      cmake --build .
      # Explicitly build Release config for multi-config generators (VS)
      # cmake --build . --config Release
      ```
   -  **Native Tools (Optional):**
      -  Linux/macOS: `make`
      -  Windows (MSBuild): `msbuild SentimentAnalysis.sln /p:Configuration=Release /p:Platform=x64` (Adjust solution/platform)

The executable `sentiment_analyzer` (or `.exe`) will be in the `build` directory or a subdirectory like `build/Release/`.

## Quick Start

### 1. Prepare Data

-  Create a CSV file (e.g., `data/sample_data.csv`) **without a header row**.
-  Format each line as: `text_content,label`
   ```csv
   I loved the movie it was great!,positive
   Terrible service very disappointed.,negative
   The weather is okay today.,neutral
   ```

### 2. Run the Application

-  Navigate to the directory containing the compiled executable (e.g., `build/`).
-  Run it, providing the relative path to your data file:

   ```bash
   # From 'build' directory (Linux/macOS)
   ./sentiment_analyzer ../data/sample_data.csv

   # From 'build/Release' directory (Windows)
   .\Release\sentiment_analyzer.exe ..\..\data\sample_data.csv
   ```

### 3. Interactive Prediction

-  After training and evaluation, the program will prompt you:
   ```
   --- Interactive Sentiment Analysis ---
   Enter text to classify (or press Enter to quit): The plot was predictable but enjoyable.
     -> Predicted Sentiment: positive
   Enter text to classify (or press Enter to quit):
   Exiting.
   ```

## Documentation

-  **Code Documentation:** Doxygen-style comments are used throughout the header files (`include/`) for classes and public methods. Generate documentation using Doxygen if needed.
-  **Technical Overview:** See the [Technical Overview](#technical-overview) section above for details on the pipeline and algorithms.
-  **Build/Run:** See [Installation](#installation) and [Quick Start](#quick-start).
-  **Components:** Refer to the [Components](#components) table.

## System Architecture

The application follows a sequential pipeline architecture implemented through modular C++ classes:

1. **Input (`DataLoader`):** Reads raw data from a file.
2. **Preprocessing (`Preprocessor`):** Cleans and tokenizes text.
3. **Feature Extraction (`FeatureExtractor`):** Builds vocabulary (training only) and converts text to Bag-of-Words vectors.
4. **Training (`NaiveBayesClassifier::train`):** Learns model parameters (priors, likelihoods) from training features and labels.
5. **Evaluation (`main.cpp` loop):** Uses the trained model (`NaiveBayesClassifier::predict`) to classify validation data and calculate accuracy.
6. **Inference (`main.cpp` loop):** Uses the trained model (`NaiveBayesClassifier::predict`) to classify new user input after applying the same preprocessing and feature extraction steps.

This modular design allows individual components to be potentially tested or replaced.

## Requirements

-  C++17 Compliant Compiler (GCC 7+, Clang 5+, MSVC 19.14+)
-  CMake (>= 3.14)
-  Git (for cloning)
-  Standard C++ Library (no external ML libraries required for core functionality)

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/YourFeature`).
3. Commit your changes (`git commit -m 'Add some feature'`). Adhere to existing code style.
4. Push to the branch (`git push origin feature/YourFeature`).
5. Open a Pull Request.

### Development Setup

```bash
# Clone your fork
git clone <your-fork-url>
cd sentiment-analysis-cpp

# Create build directory and configure (e.g., Debug)
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug

# Build
cmake --build .

# (Optional) Add tests using CTest/Google Test/Catch2 and run them
# enable_testing() # in CMakeLists.txt
# ctest # inside build directory
```
