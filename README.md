Website Fingerprinting on Tor.

---

## ▶️ Project Overview

This project explores website fingerprinting attacks on anonymous networks, specifically targeting Tor traffic patterns to classify sites based on packet data.

The attack is divided into two scenarios: ClosedWorld and OpenWorld.

1. ClosedWorld: Multi-class classification to identify 95 monitored websites.
2. OpenWorld: A multi-class classification within unmonitored traffic to identify the 95 sites and a binary classification of monitored vs. unmonitored traffic.

---

## ▶️ Project Structure

Data Source: mon_standard.pkl (Feature X1: timestamps, X2: direction*size, y: site of each instance), umnom_standard10_3000.pkl (Feature X1: timestamps, X2: direction*size)

Feature Extraction: We extract key features such as relative X1: packet timestamps, X2: packet sizes, X3: cumulative packet sizes, and X4: burst patterns to distinguish traffic between websites.

- X3 and X4 are the added features.

ClosedWorld Classification: Multi-class classification for 95 specific websites.

Model Selection: RandomForestClassifier (After trying many models such as RFC-SVC, RandomForestClassifier, XGBoost, etc., we found that RandomForestClassifier performs best.) are evaluated to determine the optimal model for this setting.

Evaluation Metrics: Accuracy, precision, recall, ROC are the metrics used for evaluating the model in the ClosedWorld scenario.

OpenWorld Binary Classification: Monitored vs. unmonitored traffic, identifying whether the traffic belongs to one of the 95 monitored websites(labeled as 1) or not(labeled as -1).

OpenWorld Multi-class Classification: Classify 95 monitored website traces with unique labels against additional unmonitored websites(labeled as -1).

Evaluation Metrics: True positive rate, false positive rate, precision-recall curve, and ROC are considered for evaluating OpenWorld results.

---

## ▶️ Experiment Settings:

### 1. Size of the data

-  mon_standard.pkl - 19000 instances(95 websites, each with 10 subpages which are non-index -  pages, observed 20 times each), features : X1-X2
-  unmon_standard10_3000.pkl - 3000 instances, features : X1-X2
-  mon_standard_withlabel.pkl - 19000 instances, features : X1-X4, label : 0-94
-  mon_standard_withlabel_binary.pkl - 19000 instances, features : X1-X4, label : 1
-  unmon_standard10_3000_withlabel.pkl - 3000 instances, features : X1-X4, label : -1
-  combined_dataset.pkl - 22000 instances, features : X1-X4, label : -1 and 0-94
-  combined_dataset_binary.pkl - 22000 instances, features : X1-X4, label : -1 and 1

### 2. Resources needed

-  RAM used : _GB
-  Available number of CPU Core : 8

### 3. Execution Environment

-  Python : 3.11.7
-  scikit-learn : 1.5.2
-  numpy : 1.26.4

---

## ▶️ How to run the model:

1. Download mon_standard.pkl and unmon_standard10_3000.pkl
2. Execute ClosedWorld_monitored.ipynb.
3. Execute OpenWorld_multiclass_classification.ipynb. Then you will get mon_standard_withlabel.pkl, unmon_standard10_3000_withlabel.pkl, and combined_dataset.pkl.
4. Execute OpenWorld_binary_classification.ipynb. Then you will get mon_standard_withlabel_binary.pkl and combined_dataset_binary.pkl.

P.S. If you want to check the original dataset, execute load_pickle_code.ipynb. This practice is not essential.

---

## ▶️ Experimental Results and Model Selection Justification

### 1. Experimental Results

- **Random Forest (Entropy)**: Accuracy = `0.9018`
- **Random Forest (Gini)**: Accuracy = `0.8981`
- **Boosted Random Forest**: Accuracy = `0.9018` (no improvement over standard Random Forest)
- **Linear SVM**: Accuracy = `0.8963`
- **Pipelined SVM**: Accuracy = `0.8844` (did not perform better compared to standalone Linear SVM)

### 2. Model and Parameter Selection Justification

- **Random Forest with Entropy** was chosen due to its high accuracy (`0.9018`) compared to other tested models.
- **Random Forest (Gini)** had slightly lower performance (`0.8981`) but was still competitive.
- **Boosted Random Forest** did not show improvement, indicating boosting was not effective for this dataset.
- **Linear SVM** performed reasonably well (`0.8963`), but not as well as Random Forest.
- **Pipelined SVM** (with scaling and hyperparameter tuning) had lower accuracy (`0.8844`), suggesting that additional complexity did not yield better results.

Overall, **Random Forest (Entropy)** was selected for its balance of accuracy and simplicity, while other models either underperformed or added unnecessary complexity.

---
