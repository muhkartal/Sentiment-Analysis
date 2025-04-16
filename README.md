# Simple C++ Sentiment Analysis

This project implements a basic sentiment analysis pipeline in C++ (17 or higher) using a Naive Bayes classifier and a Bag-of-Words feature representation. It classifies text into positive, negative, or neutral categories.

## Features

-  **Data Loading:** Loads text data and labels from a CSV file.
-  **Preprocessing:** Performs lowercase conversion, punctuation removal, and tokenization.
-  **Feature Extraction:** Uses a Bag-of-Words (frequency count) model.
-  **Classification:** Implements a Multinomial Naive Bayes classifier from scratch.
-  **Evaluation:** Splits data into training/validation sets and reports accuracy.
-  **Inference:** Allows classifying new text input via the command line after training.
-  **Build System:** Uses CMake for cross-platform building.
-  **Code Quality:** Aims for clean, documented, and well-structured C++ code.

## Dependencies

-  A C++17 compliant compiler (e.g., GCC 7+, Clang 5+, MSVC 19.14+).
-  CMake (version 3.14 or higher).

## Building the Project

1. **Clone the repository:**

   ```bash
   git clone <your-repo-url>
   cd sentiment-analysis-cpp
   ```

2. **Create a build directory:**

   ```bash
   mkdir build
   cd build
   ```

3. **Configure using CMake:**

   ```bash
   cmake ..
   ```

   -  _Optional:_ Specify a build type (e.g., for release optimizations): `cmake .. -DCMAKE_BUILD_TYPE=Release`
   -  _Optional:_ Specify a generator if needed (e.g., for Visual Studio): `cmake .. -G "Visual Studio 17 2022"`

4. **Build the project:**
   ```bash
   cmake --build .
   ```
   -  _Optional (Linux/macOS):_ Use Make directly: `make`
   -  _Optional (Windows with MSBuild):_ Use MSBuild: `msbuild SentimentAnalysis.sln /p:Configuration=Release` (Adjust solution file name if needed).

This will create an executable named `sentiment_analyzer` (or `sentiment_analyzer.exe` on Windows) in the `build` directory (or a subdirectory like `build/Debug` or `build/Release`).

## Running the Project

1. **Prepare Data:**

   -  Place your training data in the `data/` directory.
   -  The expected format is a CSV file (e.g., `data/sample_data.csv`) with **no header row**, where each line contains: `text,sentiment_label`
   -  Example `data/sample_data.csv`:
      ```csv
      I love this movie!,positive
      This was a terrible experience.,negative
      The weather today is okay.,neutral
      Absolutely fantastic product.,positive
      I'm not sure how I feel about it.,neutral
      Worst service ever.,negative
      ```
   -  You can find sentiment datasets on platforms like Kaggle or create your own.

2. **Run the Analyzer:**
   Navigate to the build directory and run the executable, providing the path to your data file:

   ```bash
   ./sentiment_analyzer ../data/sample_data.csv
   ```

   (On Windows, use backslashes and add `.exe`: `.\sentiment_analyzer.exe ..\data\sample_data.csv`)

3. **Output:**
   The program will:
   -  Load and preprocess the data.
   -  Split the data into training (80%) and validation (20%) sets.
   -  Train the Naive Bayes classifier on the training set.
   -  Evaluate the classifier on the validation set and print the accuracy.
   -  Enter an interactive loop where you can type text to classify. Enter an empty line to quit.

## Project Structure
