# 24_ML_CourseProject
Website Fingerprinting on Tor.

Experiment Settings: 

'''

How to run the model:

1. Download mon_standard.pkl and unmon_standard10_3000.pkl
2. Execute ClosedWorld_monitored.ipynb.
3. Execute OpenWorld_multiclass_classification.ipynb. Then you will get mon_standard_withlabel.pkl, unmon_standard10_3000_withlabel.pkl, and combined_dataset.pkl.
4. Execute OpenWorld_binary_classification.ipynb. Then you will get mon_standard_withlabel_binary.pkl and combined_dataset_binary.pkl.

'''


Project Overview

This project explores website fingerprinting attacks on anonymous networks, specifically targeting Tor traffic patterns to classify sites based on packet data.
The attack is divided into two scenarios: ClosedWorld and OpenWorld.

ClosedWorld: Multi-class classification to identify 95 monitored websites.
OpenWorld: A multi-class classification within unmonitored traffic to identify the 95 sites and a binary classification of monitored vs. unmonitored traffic.
'''

Project Structure


Data Processing
Data Source: Packets are captured from Tor traffic and filtered based on the client IP and Tor relay IPs.
Feature Extraction: We extract key features such as relative X1: packet timestamps, X2: packet sizes, X3: cumulative packet sizes, and X4: burst patterns to distinguish traffic between websites.
X3 and X4 are the added features.



ClosedWorld Classification
Goal: Multi-class classification for 95 specific websites.
Model Selection: RandomForestClassifier (After trying many models such as RFC-SVC, RandomForestClassifier, XGBoost, etc., we found that RandomForestClassifier performs best.) are evaluated to determine the optimal model for this setting.
Evaluation Metrics: Accuracy, precision, recall, ROC are the metrics used for evaluating the model in the ClosedWorld scenario.

OpenWorld Classification
Binary Classification: Monitored vs. unmonitored traffic, identifying whether the traffic belongs to one of the 95 monitored websites(labeled as 1) or not(labeled as -1).
Multi-class Classification: Classify 95 monitored website traces with unique labels against additional unmonitored websites(labeled as -1).
Evaluation Metrics: True positive rate, false positive rate, precision-recall curve, and ROC are considered for evaluating OpenWorld results.



Installation and Dependencies
Dependencies: Install necessary libraries by running:
The project primarily relies on libraries such as numpy, scikit-learn for data processing, model building, and evaluation.



Data Setup:
Ensure mon_standard.pkl data is structured in a .pkl format. The data should include:
X1: Relative timestamps of packets.
X2: Packet sizes.
Y: Labels corresponding to website classes for classification.

Ensure unmon_standard10_3000.pkl data is structured in a .pkl format. The data should include:
X1: Relative timestamps of packets.
X2: Packet sizes.

These features will be added on the given dataset in the .ipynb model files: [X3: Cumulative packet sizes, X4: Bursts.]
