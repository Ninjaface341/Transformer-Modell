# Project Documentation

## **📂 Project Overview:**
This project focuses on fine-tuning and evaluating transformer-based models for **Masked Language Modeling (MLM)** and **Next Sentence Prediction (NSP)** using the Hugging Face `transformers` library.

The models were trained on a combination of **BookCorpus** and **Shakespeare** texts, followed by evaluation scripts that measure the model's performance.

---

## **🔹 Requirements:**
To run this project, you will need to meet the following requirements:

- **Python 3.9**: This project is optimized for Python 3.9. Please ensure you have this version installed.
- Install the required packages listed below.

### **requirements.txt:**
```bash
torch>=1.13
transformers>=4.37.0
datasets>=2.16.0
accelerate>=0.27.0
nltk>=3.8
requests>=2.31
```
Alternatively, you can install the primary dependency directly:
```bash
pip install transformers
```
This will automatically handle most of the related dependencies.

---

## **🔹 Installation Guide:**
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-repo/transformer-evaluation.git
   cd transformer-evaluation
   ```

2. **Set Up Virtual Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *Or install the main package directly:*
   ```bash
   pip install transformers
   ```

---

## **🔹 Running the Evaluation Scripts:**

### 1. **Masked Language Modeling (MLM) Evaluation:**
To evaluate the performance of the MLM model, run the following command:

```bash
python evaluation_mlm.py
```

### 2. **Next Sentence Prediction (NSP) Evaluation:**
To evaluate the NSP model:

```bash
python evaluation_nsp.py
```

---

## **🔹 Folder Structure:**
```bash
.
├── data_preparation_mlm.py
├── data_preparation_nsp.py
├── data_preparation_combined.py
├── mlm_training.py
├── nsp_training.py
├── combined_training.py
├── evaluation_mlm.py
├── evaluation_nsp.py
├── utils.py
├── requirements.txt
└── README.md
├── licence
```
## 📥 Download Models here:
- [Masked Language Model (MLM)](https://huggingface.co/Ninja666/bookcorpus_mlm_model)
- [Next Sentence Prediction (NSP)](https://huggingface.co/Ninja666/bookcorpus-nsp-model)
- [Kombiniertes Modell](https://huggingface.co/Ninja666/bert_combined_model)
---

## **🔹 Key Features:**
- **Masked Language Modeling (MLM):** Predict masked tokens in input sentences using fine-tuned models.
- **Next Sentence Prediction (NSP):** Determine whether two sentences logically follow each other.
- **Accurate Evaluation Metrics:** Detailed accuracy calculations for both MLM and NSP tasks.

---

## **🌟 Acknowledgements:**
- This project leverages the **Hugging Face Transformers** library and datasets from **BookCorpus** and **Project Gutenberg**.
- Special thanks to the Hugging Face team for providing open-source tools for NLP research.

---

## **🔹 License:**
This project is licensed under the **Apache License 2.0**. See the LICENSE file for more details.

---

For any issues or contributions, feel free to submit a pull request or open an issue in the repository.

---

🚀

