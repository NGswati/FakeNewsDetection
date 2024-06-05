# FakeNewsDetection
Machine Learning Cultivating Trust: Unveiling Misinformation - A Machine Learning Initiative for Detecting Fake News on Social Media

## Introduction

Fake news has become a significant challenge in the digital age, particularly on social media platforms where misinformation spreads rapidly. This repository, **FakeNewsDetection**, provides a Machine Learning-based solution to detect fake news and mitigate its impact by cultivating trust in online information. The initiative leverages state-of-the-art classification techniques to analyze and categorize news articles as real or fake.

## Features

- **Multiple Classifier Models**: Implementation of various machine learning algorithms including Logistic Classifier, Passive Aggressive Classifier (PAC), and others for detecting fake news.
- **Performance Analysis**: Comprehensive evaluation of models using metrics such as accuracy, precision, recall, confusion matrices, and ROC curves.
- **Visualization Tools**: Graphs and plots like Precision-Recall curves and accuracy comparison graphs to aid in understanding model performance.
- **Automation**: Automated labeling of real and fake news data.

## Repository Structure

```
FakeNewsDetection/
├── Logistic_Classifier.py                # Logistic Regression implementation
├── Passive_Aggresive_Classifier.py       # Passive Aggressive Classifier
├── Passive_Aggresive_Classifier_CR.py    # PAC Classification Report generation
├── Precision_recall_curve.py             # Precision-Recall Curve plotting
├── ROC_curve.py                          # ROC Curve plotting
├── Real_FakeLable.py                     # Script for automated real/fake labeling
├── confusion_matrix_LogisticClassifier.py# Confusion Matrix for Logistic Classifier
├── multipleClassifier_accuracyGraph.py   # Accuracy graph for multiple classifiers
├── multipleClassifier_confusionMatrix.py # Confusion matrix comparison of classifiers
├── optimalClassifier.py                  # Script for identifying the optimal classifier
├── CMimage                               # Folder containing confusion matrix images
├── requirement.txt                       # imports and downloads to be made
├── README.md                             # Documentation
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/NGswati/FakeNewsDetection.git
   cd FakeNewsDetection
   ```
2. Install the required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Prepare your dataset by ensuring it has labeled columns for `text` and `label` (real or fake).
2. Run multiple classifier file and check accuracy for all different classifiers.
   ```bash
   python multipleClassifier_accuracyGraph.py
   ```
3. Choose the classifier you want to use, e.g., Logistic Regression or Passive Aggressive Classifier.
4. Run the respective script to train the model:
   ```bash
   python Logistic_Classifier.py
   ```
5. Visualize performance metrics using the provided visualization scripts.

## Examples

- **Confusion Matrix**: Generate and save confusion matrix plots for comparison between models.
- **Precision-Recall Curve**: Plot precision-recall curves to assess model robustness.
- **Accuracy Comparison**: Compare accuracy across multiple classifiers.

## Key Insights

- Logistic Regression emerged as the most effective classifier for this dataset, providing the best balance of accuracy and precision.
- The Passive Aggressive Classifier was used for additional analysis, offering unique advantages in handling streaming data.

## Results

The repository includes scripts to:

- Generate confusion matrices and classification reports for detailed evaluation.
- Compare the performance of multiple classifiers graphically.
- Visualize Precision-Recall and ROC curves to identify trade-offs between precision and recall.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with detailed explanations of your changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Empowering trust with Machine Learning - Together, let’s combat misinformation!

