# KYK Oda Eslestirme Optimizasyonu
### KYK Dormitory Room Matching Optimization System

An AI-powered student-room matching system for KYK (Turkish Government Dormitories) that optimizes roommate assignments based on behavioral compatibility profiles.

---

## Project Status

Status: Functional Demo -- This project was developed as a real-world solution for KYK dormitory room assignments. After presenting the system to the dormitory administration, it was determined that government-affiliated institutions are unable to adopt third-party software solutions due to regulatory constraints. The project remains as a fully functional demo showcasing the complete ML pipeline.

---

## Problem and Motivation

Assigning students to dormitory rooms is typically done randomly, often resulting in roommate conflicts. This project aims to predict compatible roommate pairings using behavioral and lifestyle data, optimizing room assignments to maximize student satisfaction.

## Architecture

The system operates in two phases:

Phase 1 (Years 1-5):              Phase 2 (Year 6+):
Rule-Based Matching (v1)           AI-Powered Matching (v2)
        |                                  |
                v                                  v
                   Min-Cost Max-Flow        Random Forest Regressor
                      Global Optimization      Trained on 5 years of
                              |                   satisfaction feedback
                                      v                          |
                                         Yearly Feedback -------> Data Accumulation
                                            Collection                      |
                                                                               v
                                                                                                          Hybrid Scoring:
                                                                                                                                     a*AI + (1-a)*Rules
                                                                                                                                     
                                                                                                                                     1. Years 1-5: Rule-based algorithm (v1) handles student placements.
                                                                                                                                     2. End of each year: Satisfaction feedback is collected from matched students.
                                                                                                                                     3. After 5 years: A Random Forest model is trained on accumulated data.
                                                                                                                                     4. Year 6+: AI-powered matching (v2) takes over with a hybrid scoring approach.
                                                                                                                                     
                                                                                                                                     ## Matching Dimensions
                                                                                                                                     
                                                                                                                                     Each student and room is profiled across 6 behavioral dimensions (scale 1-10):
                                                                                                                                     
                                                                                                                                     | Dimension | Student Feature | Room Feature |
                                                                                                                                     |---|---|---|
                                                                                                                                     | Noise | noise_tolerance | room_noise_profile |
                                                                                                                                     | Smoking | smoking_level | room_smoking_profile |
                                                                                                                                     | Environment Sensitivity | environment_sensitivity | room_environment_irritant_level |
                                                                                                                                     | Wake Time | wake_time | room_wake_profile |
                                                                                                                                     | Entry Time | entry_time | room_entry_profile |
                                                                                                                                     | Sleep Interruption | sleep_interrupt_sensitivity | (derived from room noise profile) |
                                                                                                                                     
                                                                                                                                     ## Algorithms
                                                                                                                                     
                                                                                                                                     ### v1 -- Rule-Based Matching
                                                                                                                                     - Min-Cost Max-Flow for globally optimal assignment (not greedy)
                                                                                                                                     - Hard constraint filters (smoking + environmental incompatibility)
                                                                                                                                     - Weighted similarity scoring (linear + circular distance)
                                                                                                                                     
                                                                                                                                     ### v2 -- AI-Powered Matching
                                                                                                                                     - Random Forest Regressor (scikit-learn, 200 estimators)
                                                                                                                                     - 5-fold cross-validation for model evaluation
                                                                                                                                     - Hybrid mode: alpha * AI_score + (1-alpha) * rule_score for gradual transition
                                                                                                                                     - Feature importance analysis for interpretable results
                                                                                                                                     
                                                                                                                                     ## File Structure
                                                                                                                                     
                                                                                                                                     .
                                                                                                                                     +-- kyk_matcher_v1.py       # Rule-based matching algorithm
                                                                                                                                     +-- kyk_matcher_ui.py       # v1 Tkinter GUI
                                                                                                                                     +-- kyk_matcher_v2.py       # AI-powered matching
                                                                                                                                     +-- kyk_matcher_v2_ui.py    # v2 Tkinter GUI
                                                                                                                                     +-- ai_model.py             # Random Forest model (train/predict)
                                                                                                                                     +-- generate_feedback.py    # Synthetic feedback data generator
                                                                                                                                     +-- pipeline.py             # End-to-end pipeline
                                                                                                                                     +-- rooms.csv               # Sample room data
                                                                                                                                     +-- new_students.csv        # Sample student data
                                                                                                                                     +-- assignments.csv         # v1 matching results (sample)
                                                                                                                                     +-- requirements.txt        # Python dependencies
                                                                                                                                     +-- README.md
                                                                                                                                     
                                                                                                                                     ## Getting Started
                                                                                                                                     
                                                                                                                                     ### Installation
                                                                                                                                     pip install -r requirements.txt
                                                                                                                                     
                                                                                                                                     ### Quick Start (Full Pipeline)
                                                                                                                                     Run all steps sequentially:
                                                                                                                                     python pipeline.py
                                                                                                                                     
                                                                                                                                     ### v1 -- Rule-Based Matching
                                                                                                                                     python kyk_matcher_v1.py        # Command line
                                                                                                                                     python kyk_matcher_ui.py        # GUI
                                                                                                                                     
                                                                                                                                     ### v2 -- AI-Powered Matching
                                                                                                                                     python kyk_matcher_v2.py        # Command line
                                                                                                                                     python kyk_matcher_v2_ui.py     # GUI
                                                                                                                                     
                                                                                                                                     ### v2 GUI Usage Steps
                                                                                                                                     1. Load Room CSV -- Select the CSV file containing room data
                                                                                                                                     2. Load Student CSV -- Select new student data
                                                                                                                                     3. Generate/Load Feedback -- Generate synthetic data or load existing feedback
                                                                                                                                     4. Train Model -- Train the Random Forest model
                                                                                                                                     5. v2 Match -- Run AI-powered matching (adjust AI weight with alpha slider)
                                                                                                                                     6. Save Results -- Export to CSV
                                                                                                                                     
                                                                                                                                     ## Requirements
                                                                                                                                     
                                                                                                                                     - Python 3.8+
                                                                                                                                     - pandas
                                                                                                                                     - scikit-learn
                                                                                                                                     - numpy
                                                                                                                                     - joblib
                                                                                                                                     - tkinter (included with Python)
                                                                                                                                     
                                                                                                                                     ## Tech Stack
                                                                                                                                     
                                                                                                                                     Python . scikit-learn . pandas . NumPy . Tkinter . Random Forest . Min-Cost Max-Flow
                                                                                                                                     
                                                                                                                                     ## License
                                                                                                                                     
                                                                                                                                     This project is for educational and demonstration purposes.
                                                                                                                                     
