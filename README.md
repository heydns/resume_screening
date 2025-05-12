# AI-Powered Resume Screening System

This project explores a novel machine learning approach to automated resume screening using semantic matching rather than traditional classification or keyword filtering. By fine-tuning a dual-encoder architecture with contrastive learning and MarginMSELoss, the system aims to retrieve the most relevant resumes for a given job description without relying on supervised labels.

## 🧠 Project Summary

Traditional resume screening methods suffer from issues like keyword dependence, inefficiency, and lack of contextual understanding. This project reframes resume screening as a semantic retrieval task using modern NLP techniques. Instead of classifying resumes or relying on rules, we measure their semantic closeness to a job description using dense embeddings and contrastive training.

## 📁 Project Structure

```
Resume Screening Project/
│
├── Research Report.docx               # Final report for academic submission
├── Human Evaluation Information      # Doc summarizing candidate rankings by 5 humans for the script 04
├── requirements.txt                  # Python dependencies
├── dual_encoder_full_model/         # Fine-tuned SentenceTransformer model
│
├── data/
│   ├── UpdatedResumeDataSet.csv         # Original resume dataset (900+ resumes)
│   ├── full_resume_dataset_with_queries.csv # Resume+query pairs
│   ├── final_triplets_for_training.csv  # Cleaned contrastive learning triplets
│   ├── triplets_with_crossencoder_scores.csv
│
├── scripts/
│   ├── 00_full_pipeline_train.py       # Full training pipeline (query gen, mining, training)
│   ├── 01_evaluate_simulated_queries.py # MRR, Recall@K evaluation across categories
│   ├── 02_compare_model_to_baseline.py  # Evaluation vs baseline (MiniLM)
│   ├── 03_compare_model_to_human.py     # Human ranking vs model (Kendall Tau, Spearman Rho)
│   ├── 04_rank_cvs_given_job_descriptions.py  # Run ranking on new job/CV pairs
```

## 🚀 How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Make sure you have a working OpenAI API key set if using GPT-based query generation.

### 2. Train the Model (if not already trained)

```bash
python scripts/00_full_pipeline_train.py
```

This includes:
- Query generation (GPT-3.5)
- Hard negative mining
- Cross-encoder scoring
- MarginMSE contrastive training

### 3. Evaluate the Model

```bash
python scripts/01_evaluate_simulated_queries.py
python scripts/02_compare_model_to_baseline.py
```

### 4. Human Evaluation

```bash
python scripts/03_compare_model_to_human.py
```

### 5. Rank New CVs for a Job Description

```bash
python scripts/04_rank_cvs_given_job_descriptions.py
```

Edit the job description and CVs inside the script for custom inputs.

## 📊 Key Results

| Metric           | Fine-Tuned Model | MiniLM Baseline |
|------------------|------------------|------------------|
| MRR              | 0.96             | 0.91             |
| Recall@1         | 92%              | 84%              |
| Recall@5         | 100%             | 100%             |

**Human Evaluation:**
- Avg. Kendall Tau: 0.72
- Avg. Spearman Rho: 0.86

## 📚 Techniques Used

- Dual Encoder architecture (Sentence Transformers)
- MarginMSELoss contrastive learning
- Hard negative mining
- GPT-3.5-based pseudo-query generation
- Cosine similarity-based retrieval
- Evaluation with IR metrics + human judgment

## 📌 Limitations & Future Work

- Evaluation on human relevance is limited to 5 raters. Statistical significance will require broader participation.
- Current system is CLI-based; a web UI may enhance usability.
- Bias mitigation is an emergent property but not explicitly evaluated—future work can explore fairness metrics and auditability.

## 📄 License

For academic and educational use only. Contact: dubreuilkenzo@gmail.com
