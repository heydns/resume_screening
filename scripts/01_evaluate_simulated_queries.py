import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import random

# === CONFIG ===
MODEL_PATH = "../dual_encoder_full_model"
CSV_PATH = "../data/UpdatedResumeDataSet.csv"
K = 5  # top-k to evaluate Recall@k

# === Load data ===
print("🔹 Loading resumes...")
df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.lower().str.strip()
df = df.rename(columns={"resume": "text", "category": "category"})
df.dropna(subset=["text", "category"], inplace=True)

# === Load model ===
print("🔹 Loading model...")
model = SentenceTransformer(MODEL_PATH)

# === Group resumes by category ===
category_to_resumes = {}
for _, row in df.iterrows():
    cat = row["category"]
    if cat not in category_to_resumes:
        category_to_resumes[cat] = []
    category_to_resumes[cat].append(row["text"])

categories = list(category_to_resumes.keys())
print(f"✅ Found {len(categories)} unique categories.")

# === Evaluation ===
mrr_total = 0
recall_at_1 = 0
recall_at_k = 0
hits_at_rank = [0] * K
total_queries = 0

print("\n🚀 Evaluating simulated job queries...\n")
for cat in tqdm(categories, desc="Categories"):
    positives = category_to_resumes[cat]
    if len(positives) < 1:
        continue

    # Pick one resume from the category as query
    query_text = f"{cat} job description"  # simulated query
    positive_resume = random.choice(positives)

    # Pick 10 negatives from other categories
    negatives = []
    other_cats = [c for c in categories if c != cat]
    while len(negatives) < 10:
        neg_cat = random.choice(other_cats)
        neg_resumes = category_to_resumes[neg_cat]
        if neg_resumes:
            negatives.append(random.choice(neg_resumes))

    # Build corpus (1 positive + 10 negatives)
    candidates = [positive_resume] + negatives
    labels = [1] + [0] * len(negatives)

    # Encode
    query_emb = model.encode(query_text, convert_to_tensor=True)
    cand_embs = model.encode(candidates, convert_to_tensor=True)

    # Compute similarities
    scores = util.cos_sim(query_emb, cand_embs)[0]
    sorted_indices = torch.argsort(scores, descending=True)

    # Compute metrics
    total_queries += 1
    for rank, idx in enumerate(sorted_indices[:K]):
        if labels[idx] == 1:
            mrr_total += 1 / (rank + 1)
            hits_at_rank[rank] += 1
            if rank == 0:
                recall_at_1 += 1
            if rank < K:
                recall_at_k += 1
            break

# === Results ===
print("\n✅ Evaluation Results:")
print(f"MRR:        {mrr_total / total_queries:.4f}")
print(f"Recall@1:   {recall_at_1 / total_queries:.4f}")
print(f"Recall@{K}: {recall_at_k / total_queries:.4f}")

print("\n📊 Hits by Rank Position:")
for i, count in enumerate(hits_at_rank):
    print(f"  Rank {i+1}: {count} matches ({100 * count / total_queries:.2f}%)")
