Final Result
Media: Class Distribution
•	Classification Categories: Established four-class system: 
o	Class 0: Normal (healthy heart sounds) had 351 audio files.
o	Class 1: Murmur (abnormal whooshing sounds) had 129 audio files.
o	Class 2: Artifact (non-cardiac noise) had 40 audio files.
o	Class 3: Extra heart sounds (additional sounds beyond S1/S2) had 19 audio files.
Media: Bio-medical signal processing
<img width="804" height="536" alt="image" src="https://github.com/user-attachments/assets/2ec013e4-0e8e-4d5e-a8d9-00420fbca53f" />

Initial Baseline Classification Modelling
Performance Comparison:
Model	Training Time (s)	Inference Time (s)	Accuracy	Abnormal Recall	False Negative Rate
SVM (Default)	25.97	8.51	0.78	0.55	0.45
SVM (Grid)	704.21	9.89	0.85	0.76	0.24
RF (Default)	3.22	0.034	0.82	0.61	0.39
RF (Grid)	56.65	0.061	0.82	0.60	0.40
XGB (Default)	4.03	0.011	0.85	0.71	0.29
XGB (Grid)	99.40	0.011	0.87	0.71	0.29

Media: Performance Comparison for Accuracy, Recall, and FNR after Grid Search CV
<img width="970" height="611" alt="image" src="https://github.com/user-attachments/assets/117e9982-12dd-4a26-b0f0-fe1f5f415321" />

Media: Pareto Front Analysis after Multi Objective Bayesian Optimization
<img width="969" height="726" alt="image" src="https://github.com/user-attachments/assets/a19b6b46-12e8-445c-bfbe-9a2684a4b869" />

Computational Performance:
•	Total Optimization Time: 3.88 hours (13,977 seconds)
•	Total Evaluations: 900 configurations
•	Algorithm Distribution: XG Boost (74.1%), SVM (15.0%), Random Forest (10.9%)
Best Individual Metrics:
Accuracy	88.43%	XGBoost	Evaluation #567
Abnormal Recall	83.29%	SVM	Evaluation #340
F1 Score	88.29%	SVM	Evaluation #340
Lowest FNR	16.71%	SVM	Evaluation #340
Fastest Training	1.13s	Random Forest	Optimized config
Fastest Testing	0.014s	Random Forest	Optimized config
