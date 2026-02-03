# Potato Leaf Disease Detection with Deep Learning + Big Data (Hadoop & Spark)

📖 **Overview**

This repository contains the implementation and supporting materials for my MSc Data Analytics Continuous Assessment project at CCT College Dublin (Semester 2). The project investigates the classification of potato leaf diseases using deep learning models and evaluates how different architectures perform when integrated with distributed data storage (HDFS) and parallel data processing (Apache Spark).

The models implemented are:
- **Custom CNN**
- **VGG-16**
- **MobileNetV2**

The dataset is stored and managed using **Hadoop Distributed File System (HDFS)** and processed with **Apache Spark**.

---

🎯 **Research Objectives**

The main goals of this project are:

- To compare the performance of **Custom CNN**, **VGG-16**, and **MobileNetV2** in classifying potato leaf diseases.
- To demonstrate how **Hadoop** can be used for scalable dataset storage.
- To explore how **Apache Spark** accelerates data preprocessing and loading pipelines.
- To evaluate model prediction performance using standard classification metrics (accuracy, precision, recall, F1-score).

---

🗂 **Dataset**

- **Source:** Kaggle potato leaf disease dataset  
- **Total Images:** 2152  
- **Classes:**
  - Early Blight
  - Late Blight
  - Healthy  
- **Input Size:** 224 × 224 RGB  
- **License:** Public Domain / CC0 (as stated on dataset platform)

> Note: Due to dataset size, the full image folder is not included in this repo. Please download and upload it to HDFS before running the notebooks.

---

⚙️ **Methodology**

### 1. Data Storage (Hadoop HDFS)
Images are uploaded and stored in HDFS to enable distributed storage and scalable access.

Example commands:
```bash
hdfs dfs -mkdir -p /user1/potato_disease
hdfs dfs -put Potato___Early_blight /user1/potato_disease
hdfs dfs -put Potato___Late_blight /user1/potato_disease
hdfs dfs -put Potato___healthy /user1/potato_disease
hdfs dfs -ls /user1/potato_disease
```
### 2) Data Loading (Spark)
- A Spark Session was created with:
  - `executor memory: 8G`
  - `driver memory: 8G`
- Images were loaded from HDFS into Spark DataFrames

### 3) Data Preprocessing
- Label encoding:
  - Early Blight → 0  
  - Late Blight → 1  
  - Healthy → 2  
- Resize to 224×224  
- Normalisation:
  - mean = [0.485, 0.456, 0.406]
  - std  = [0.229, 0.224, 0.225]
- Data augmentation:
  - Random horizontal flip (50%)
  - Random rotation (10°)
- Split:
  - Train: 80%
  - Test: 20%
- Batch size: 32

### 4) Models Implemented
- **Custom CNN**
  - 2 convolution layers (16 and 32 filters)
  - pooling layers + dropout + fully connected layers
- **VGG-16**
  - pretrained VGG-16
  - convolution layers frozen to reduce overfitting
- **MobileNetV2**
  - pretrained MobileNetV2
  - convolution layers frozen to reduce overfitting

### 5) Training & Evaluation
- **Epochs:** 10 (CPU training limitation)  
- **Optimizer:** Adam (lr = 1e-4)  
- **Loss:** Cross-Entropy  
- **Metrics:** Accuracy, Precision, Recall, F1-score, Confusion Matrix  

---
### 🚀 How to Open & Run the Project

### 1. Clone the Repository
```bash
git clone https://github.com/madinasagatova/Potato-Leaf-Disease-Detection.git
cd Potato-Leaf-Disease-Detection
``` 
### 2. Create & Activate Virtual Environment (recommended)
```bash
python3 -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
```
### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

If you don’t have a requirements file yet, generate one with:
```bash
pip freeze > requirements.txt
```
### 4. Upload Dataset to HDFS (example)
```bash
hdfs dfs -mkdir -p /user1/potato_disease
hdfs dfs -put <local_path_to_images> /user1/potato_disease
```
### 5. Run the Notebooks

Open Jupyter Notebook:

jupyter notebook


Then open and run, in order:

CA1Sem2MadinaSagatova2021255.ipynb

CNN_Model.ipynb

VGG16_model.ipynb

MobiletNetV2_model.ipynb

### 📑 License

This academic project uses datasets under the CC0 Public Domain license (dataset source).
Code in this repository is provided for academic and research use.

### 👩‍💻 Author

Madina Sagatova
MSc Data Analytics, CCT College Dublin (2025)

GitHub: https://github.com/madinasagatova


---
