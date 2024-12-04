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

Feature Extraction: We extract key features such as relative X1: packet timestamps, X2: packet sizes, X3: cumulative packet sizes, X4: burst patterns, X5: Time differences between packets, and X6: Standard Deviation of packets to distinguish traffic between websites.

- X3, X4, X5, and X6 are the added features.

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
-  mon_standard_withlabel.pkl - 19000 instances, features : X1-X2, label : 0-94
-  mon_standard_withlabel_binary.pkl - 19000 instances, features : X1-X2, label : 1
-  unmon_standard10_3000_withlabel.pkl - 3000 instances, features : X1-X2, label : -1
-  combined_dataset.pkl - 22000 instances, features : X1-X6, label : -1 and 0-94
-  combined_dataset_binary.pkl - 22000 instances, features : X1-X4, label : -1 and 1

### 2. Resources needed

-  RAM used : CW 6.22GB
-  Available number of CPU Core : 8

### 3. Execution Environment

-  Python : 3.11.7
-  scikit-learn : 1.5.2
-  numpy : 1.26.4

---

## ▶️ How to run the model:

1. Clone the repository to your local machine using the following command:
   `git clone https://github.com/Leeseoyeon12rl/24_ML_CourseProject.git`
2. Execute ClosedWorld_monitored.ipynb.
3. Execute OpenWorld_multiclass_classification.ipynb. Then you will get mon_standard_withlabel.pkl, unmon_standard10_3000_withlabel.pkl, and combined_dataset.pkl in db.
4. Execute OpenWorld_binary_classification.ipynb. Then you will get mon_standard_withlabel_binary.pkl and combined_dataset_binary.pkl in db.

P.S. If you want to check the original dataset, execute load_pickle_code.ipynb. This practice is not essential.

---

## ▶️ Experimental Results and Model Selection Justification

### **1. Experimental Results**

#### **ClosedWorld Results**

**Classification Metrics:**
- **Accuracy**: `0.90` (Random Forest with Entropy)  
- **Macro Avg Precision**: `0.90`, **Macro Avg Recall**: `0.90`, **Macro Avg F1-Score**: `0.90`  
- **Weighted Avg Precision**: `0.91`, **Weighted Avg Recall**: `0.90`, **Weighted Avg F1-Score**: `0.90`  

**Class-Level Performance:**
- Most classes showed F1-scores above `0.85`, demonstrating strong generalization across the dataset.
- Certain classes, such as Class `9` (F1: `0.58`), exhibited lower performance, likely due to overlapping feature patterns.

**Model Comparison:**
- **Random Forest (Entropy)**:  
  - Accuracy = `0.9018`  
  - Outperformed other models in both overall accuracy and class-level F1-scores.  
- **Random Forest (Gini)**:  
  - Accuracy = `0.8981`  
  - Similar performance to Entropy but slightly lower precision and recall.  
- **Boosted Random Forest**:  
  - Accuracy = `0.9018` (no improvement)  
  - Boosting added computational cost without tangible accuracy gains.  
- **Linear SVM**:  
  - Accuracy = `0.8963`  
  - Slightly weaker compared to Random Forest.  
- **Pipelined SVM**:  
  - Accuracy = `0.8844`  
  - Despite additional preprocessing, performance was weaker, indicating the dataset's structure is better suited for tree-based models.

![closedworldplot](https://github.com/user-attachments/assets/fdba0da3-59b3-4e1d-87b8-837318da2a89)

#### **OpenWorld Results**

**Classification Metrics:**
- **Accuracy**: `0.43` (Random Forest with Entropy)
- **Macro Avg Precision**: `0.38`, **Macro Avg Recall**: `0.36`, **Macro Avg F1-Score**: `0.36`
- **Weighted Avg Precision**: `0.41`, **Weighted Avg Recall**: `0.43`, **Weighted Avg F1-Score**: `0.40`

**Class-Level Performance:**
- The unmonitored class (`-1`) had the best performance with an F1-score of `0.70`, indicating that distinguishing between monitored and unmonitored traffic was generally successful.
- Many monitored classes had significantly lower F1-scores, indicating poor performance in fine-grained classification. For instance, Class `1` had an F1-score of `0.20`, highlighting challenges in differentiating specific monitored websites.
- Performance varied substantially among classes, with F1-scores ranging from `0.08` (Class `38`) to `0.86` (Class `75`).

**Model Insights:**
- **Random Forest (Entropy)** was selected due to its ability to handle the complexity of the OpenWorld dataset, though the overall accuracy remained relatively low (`0.43`).
- The **imbalanced dataset** contributed to the challenges observed in precision and recall metrics, particularly for the smaller classes. More effective balancing or a different model may be needed to improve these metrics.
- The current model shows potential in detecting the broader difference between monitored and unmonitored traffic but struggles in accurately identifying individual monitored websites.

**Performance Summary:**
- The **Precision-Recall Curve** analysis indicates inconsistent performance, especially across multiple thresholds. The **AUC** (`0.29`) further emphasizes the limitations in distinguishing between classes effectively.
- Overall, while **Random Forest** provided a starting point for the classification task, further optimization and possible use of different algorithms might be necessary to improve the results.

- ![image](https://github.com/user-attachments/assets/0fb68dcf-d62d-470a-9598-d74763ad527f)


#### **OpenWorld-Binary Results**

**Classification Metrics:**
- **Accuracy**: `0.9759` (Logistic Regression)  
- **High Accuracy**: Logistic Regression achieved a high accuracy of `0.9759` for distinguishing between monitored and unmonitored traffic.
- Given the satisfactory results, no additional tuning was performed for the OpenWorld-Binary scenario.

### **2. Model and Parameter Selection Justification**

#### **ClosedWorld**
- **Selected Model**: Random Forest with Entropy  
  - **Reason**: Achieved the highest overall accuracy (`0.9018`) and macro-average F1-score (`0.90`).  
  - Consistent performance across most classes.

#### **OpenWorld-Multiclass**
- **Selected Model**: Random Forest  
  - **Reason**: Random Forest was chosen for its ability to manage class imbalance and handle large datasets. The final model (n_estimators=300, max_depth=10) balanced efficiency and performance.
  - The imbalanced dataset led to difficulties in precision and recall, suggesting the need for better balancing or different models like Gradient Boosting.
  - Further tuning or different approaches are needed to improve fine-grained classification of monitored classes.


#### **OpenWorld-Binary**
- **Selected Model**: Logistic Regression  
  - **Reason**: Achieved high accuracy (`0.9759`) for distinguishing between monitored and unmonitored traffic.  
  - Given the satisfactory initial results, no additional parameter tuning was deemed necessary.

---
