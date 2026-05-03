# KYK Dormitory Room Matching Optimization System

## Project Status
This project was developed as a real-world solution for KYK dormitory room assignments. After presenting the system to the dormitory administration, it was determined that government-affiliated institutions are unable to adopt third-party software solutions due to regulatory constraints. The project remains as a fully functional demo.

## Problem and Motivation
The current KYK room assignment process often lacks optimization based on student lifestyles and preferences. This project aims to minimize friction between roommates and maximize overall satisfaction by using a data-driven approach.

## Architecture
The system operates in two phases:
1. Rule-Based Matching (v1): Uses a Min-Cost Max-Flow algorithm for globally optimal assignment (not greedy).
2. 2. AI-Powered Matching (v2): Utilizes a Random Forest Regressor and a Hybrid Scoring Mode for gradual transition from rule-based to data-driven matching.
  
   3. ## Tech Stack
   4. * Language: Python 3.8+
      * * Algorithms: Min-Cost Max-Flow (NetworkX), Random Forest (Scikit-learn)
        * * GUI: Tkinter
          * 
