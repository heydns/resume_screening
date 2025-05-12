import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from collections import defaultdict
from tqdm import tqdm
import random

# === Settings ===
CV_FILE = "../data/UpdatedResumeDataSet.csv"
MODELS = [
    ("Your Fine-Tuned Model", "../dual_encoder_full_model"),
    ("MiniLM Baseline", "sentence-transformers/all-MiniLM-L6-v2"),
]
NUM_PER_CATEGORY = 5

# === Load Data ===
print("\U0001F539 Loading resumes...")
df = pd.read_csv(CV_FILE)
df.columns = df.columns.str.lower().str.strip()
df = df.rename(columns={"resume": "text", "category": "category"})
df.dropna(subset=["text", "category"], inplace=True)

categories = df["category"].unique().tolist()
print(f"✅ Found {len(categories)} unique categories.")

# === Evaluation Function ===
def evaluate_model(model_name):
    print(f"\n\U0001F50D Evaluating model: {model_name}")
    model = SentenceTransformer(model_name)

    # Encode all resumes
    all_cv_texts = df["text"].tolist()
    all_cv_embeddings = model.encode(all_cv_texts, convert_to_tensor=True, show_progress_bar=True)

    rank_hits = defaultdict(int)
    mrr_total = 0
    recall_at_1 = 0
    recall_at_5 = 0
    total = 0

    for category in tqdm(categories, desc="Evaluating categories"):
        pos_df = df[df["category"] == category]
        neg_df = df[df["category"] != category]

        if len(pos_df) == 0 or len(neg_df) < NUM_PER_CATEGORY:
            continue

        pos_resume = pos_df.sample(n=1).iloc[0]
        neg_resumes = neg_df.sample(n=NUM_PER_CATEGORY - 1)

        candidates = pd.concat([pd.DataFrame([pos_resume]), neg_resumes]).sample(frac=1).reset_index(drop=True)
        candidate_embeddings = model.encode(candidates["text"].tolist(), convert_to_tensor=True)

        query = f"Looking to hire someone in {category}"
        query_embedding = model.encode(query, convert_to_tensor=True)

        scores = util.cos_sim(query_embedding, candidate_embeddings)[0]
        top_indices = torch.topk(scores, k=NUM_PER_CATEGORY).indices.cpu().numpy()
        top_cats = candidates.iloc[top_indices]["category"].values

        found = False
        for rank, cat in enumerate(top_cats, start=1):
            if cat == category:
                mrr_total += 1 / rank
                rank_hits[rank] += 1
                found = True
                if rank == 1:
                    recall_at_1 += 1
                recall_at_5 += 1
                break

        total += 1

    # === Report ===
    print("\n\u2705 Evaluation Results:")
    print(f"MRR:        {mrr_total / total:.4f}")
    print(f"Recall@1:   {recall_at_1 / total:.4f}")
    print(f"Recall@5:   {recall_at_5 / total:.4f}\n")

    print("\U0001F4CA Hits by Rank Position:")
    for i in range(1, NUM_PER_CATEGORY + 1):
        hits = rank_hits[i]
        print(f"  Rank {i}: {hits} matches ({hits / total:.2%})")

# === Run Comparison ===
for name, path in MODELS:
    evaluate_model(path)