Large-Scale Cybersecurity Threat Detection using PySpark & Deep Learning
Overview
This project implements a scalable intrusion detection system (IDS) using PySpark and advanced Machine Learning algorithms on the UNSW-NB15 dataset. The goal is to address the limitations of traditional IDS in detecting sophisticated cyberattacks and managing large-scale network traffic in real-time.
Key features include:
•	Distributed data processing using PySpark for scalability.
•	Feature engineering and selection to enhance model efficiency.
•	Model training using multiple ML algorithms, including Gradient Boosting and Deep Q-Network (DQN).
•	High detection accuracy (up to 96%) and false positive reduction.
•	Support for real-time adaptability in evolving threat environments.
 
Problem Statement
Traditional IDS face:
•	High false-positive rates.
•	Scalability issues with large network data.
•	Inability to detect complex, evolving attacks in real time.
This project addresses these issues using distributed ML models for large-scale, accurate, and adaptive threat detection.
 
Dataset
•	UNSW-NB15 Dataset
o	Contains 2.5M+ network traffic records.
o	Includes 49 features across normal and 9 attack types:
	Fuzzers, Analysis, Backdoors, DoS, Exploits, Generic, Reconnaissance, Shellcode, Worms.
•	Download: UNSW-NB15 Dataset
 
Architecture
1.	Data Preprocessing
	Handle missing values, normalize data.
	Encode categorical variables using StringIndexer and OneHotEncoder.
 Use VectorAssembler and StandardScaler for feature transformation.
2.	Feature Selection
	Correlation analysis & PCA (optional).
oTop features: dur, sbytes, dbytes, rate, sttl.
3.	Model Development
PySpark MLlib for distributed model training.
Algorithms:
Decision Tree
Random Forest
Naïve Bayes
Gradient Boosting (best performance)
DQN-based deep learning for adaptive detection.
5.	Performance Evaluation
Metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC.
Gradient Boosting: 94% accuracy.
DQN: 96% accuracy, reduced false positives.
 
Tech Stack
•	Programming Language: Python
•	Frameworks: PySpark (MLlib), TensorFlow (for DQN)
•	Big Data Tools: Apache Spark
•	Dataset: UNSW-NB15
•	Visualization: Matplotlib, Seaborn
 
Results
Model	Accuracy
Gradient Boosting	94%
Random Forest	90%
Decision Tree	93%
Naïve Bayes	77%
DQN	96%
•	Gradient Boosting: Best traditional ML model.
•	DQN: Achieved state-of-the-art performance and dynamic adaptability.
 
Future Work
•	Hyperparameter optimization using Grid Search and Bayesian Tuning.
•	Integrating deep learning models (LSTM, CNN) for complex pattern detection.
•	Deploying the system for real-time monitoring.
•	Adding federated learning for distributed security intelligence.
•	Exploring adversarial robustness against poisoning and evasion attacks.
 
How to Run
1.	Clone the repository
bash
CopyEdit
git clone https://github.com/yourusername/Cybersecurity-Threat-Detection.git
cd Cybersecurity-Threat-Detection
2.	Install dependencies
bash
CopyEdit
pip install -r requirements.txt
3.	Run PySpark pipeline
bash
CopyEdit
spark-submit main_pipeline.py
4.	Execute DQN model
bash
CopyEdit
jupyter notebook Final\ Code-DQN.ipynb
 
Contributors
•	Mohan Krishna Thiriveedhi
•	Pavan Chowdary Chilukuri
•	Triveni Kandimalla
•	Raghava Sammeta
•	Venkata Basanth Challapalli

