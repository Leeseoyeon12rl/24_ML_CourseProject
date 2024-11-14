# 24_ML_CourseProject
Website Fingerprinting on Tor.

'''

Project Overview

This project explores website fingerprinting attacks on anonymous networks, specifically targeting Tor traffic patterns to classify sites based on packet data.
The attack is divided into two scenarios: ClosedWorld and OpenWorld.

ClosedWorld: Multi-class classification to identify 95 monitored websites.

OpenWorld: A binary classification of monitored vs. unmonitored traffic and a subsequent multi-class classification within unmonitored traffic to identify the 95 sites.

The data is based on packet captures, leveraging packet sizes and timestamps as features for fingerprinting analysis.

'''

Project Structure


Data Processing

Data Source: Packets are captured from Tor traffic and filtered based on the client IP and Tor relay IPs.

Feature Extraction: We extract key features such as relative packet timestamps, packet sizes, cumulative packet sizes, and burst patterns to distinguish traffic between websites.



ClosedWorld Classification

Goal: Multi-class classification for 95 specific websites.

Model Selection: Various models (logistic regression, SVM, RandomForestClassifier, etc.) are evaluated to determine the optimal model for this setting.

Evaluation Metrics: Accuracy is the primary metric used for evaluating the model in the ClosedWorld scenario.

OpenWorld Classification

Binary Classification: Monitored vs. unmonitored traffic, identifying whether the traffic belongs to one of the 95 monitored websites or not.

Subsequent Multi-class Classification: For unmonitored traffic, we perform a secondary classification to identify one of the 95 websites.

Evaluation Metrics: True positive rate, false positive rate, precision-recall curve, and ROC are considered for evaluating OpenWorld results.



Installation and Dependencies

Dependencies: Install necessary libraries by running:

The project primarily relies on libraries such as numpy, scikit-learn, and tensorflow/keras for data processing, model building, and evaluation.



Data Setup: Ensure packet data is structured in a .pkl format as data.pkl. The data should include:

X1: Relative timestamps of packets.

X2: Packet sizes.

X3: Cumulative packet sizes.

X4: Packet bursts.

Y: Labels corresponding to website classes for classification.


Configure the settings within the script to select ClosedWorld or OpenWorld modes.

ClosedWorld Mode: Executes a multi-class classification over 95 monitored websites.
Results are saved under results/closedworld.

OpenWorld Mode: Runs a binary classification followed by a multi-class classification within unmonitored traffic.
Results are saved under results/openworld.
