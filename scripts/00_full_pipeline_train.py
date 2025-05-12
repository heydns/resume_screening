import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder, InputExample, util, losses
from torch.utils.data import DataLoader
import torch
import openai
import os
from tqdm import tqdm

# CONFIG
openai.api_key = os.getenv("OPENAI_API_KEY")  # 🔥 Put your OpenAI key here
input_resume_csv = "../data/UpdatedResumeDataSet.csv"  # 🔥 Your full dataset
output_dir = "../dual_encoder_full_model"      # 🔥 Where to save the trained model
batch_size = 8
epochs = 3

# 1. Load Resumes
print("🔹 Loading resumes...")
df = pd.read_csv(input_resume_csv)
assert 'Resume' in df.columns and 'Category' in df.columns, "CSV must have Resume and Category columns."

resumes = df['Resume'].tolist()
categories = df['Category'].tolist()


# 2. Generate Queries
if os.path.exists("full_resume_dataset_with_queries.csv"):
    print("🔹 Queries already generated, loading existing file...")
    df = pd.read_csv("full_resume_dataset_with_queries.csv")
    resumes = df['Resume'].tolist()
    categories = df['Category'].tolist()
    queries = df['Query'].tolist()
else:
    print("🔹 Generating queries using OpenAI...")
    queries = []
    for text in tqdm(resumes):
        prompt = f"Generate 3 short generic questions that could be asked based on the following paragraph:\n\nParagraph: {text}\n\nQuestions:"
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates questions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=150
        )
        query_text = response.choices[0].message.content
        queries.append(query_text.split('\n')[0])  # Take only the first query to simplify

    df['Query'] = queries
    df.to_csv("../data/full_resume_dataset_with_queries.csv", index=False)

# 3. Negative Mining
print("🔹 Performing dense retrieval negative mining...")
retriever_model = SentenceTransformer("all-MiniLM-L6-v2")
resume_embeddings = retriever_model.encode(resumes, convert_to_tensor=True, show_progress_bar=True)

triplets = []
for i, (query, true_resume, true_category) in enumerate(zip(queries, resumes, categories)):
    query_emb = retriever_model.encode(query, convert_to_tensor=True)
    cosine_scores = util.cos_sim(query_emb, resume_embeddings)[0]
    mask = [(idx != i and categories[idx] != true_category) for idx in range(len(resumes))]
    masked_scores = torch.where(
    torch.tensor(mask, device=cosine_scores.device),    
    cosine_scores,
    torch.tensor(-1.0, device=cosine_scores.device)
    )

    top_k = torch.topk(masked_scores, k=10)

    selected = 0
    for score, idx in zip(top_k.values, top_k.indices):
        if score.item() >= 0.25:
            triplets.append({
                "query": query,
                "positive_resume": true_resume,
                "positive_category": true_category,
                "negative_resume": resumes[idx],
                "negative_category": categories[idx],
                "negative_score": score.item()
            })
            selected += 1
        if selected == 3:
            break

triplet_df = pd.DataFrame(triplets)
triplet_df.to_csv("../data/triplets_with_category_filtered_negatives.csv", index=False)

# 4. Cross-Encoder Scoring
print("🔹 Scoring with Cross-Encoder...")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

queries = triplet_df["query"].tolist()
positives = triplet_df["positive_resume"].tolist()
negatives = triplet_df["negative_resume"].tolist()

pos_scores = []
neg_scores = []

for query, pos, neg in tqdm(zip(queries, positives, negatives), total=len(queries)):
    pos_score = cross_encoder.predict([(query, pos)])[0]
    neg_score = cross_encoder.predict([(query, neg)])[0]
    pos_scores.append(pos_score)
    neg_scores.append(neg_score)

triplet_df["pos_score"] = pos_scores
triplet_df["neg_score"] = neg_scores
triplet_df.to_csv("../data/triplets_with_crossencoder_scores.csv", index=False)

# 5. Filter Triplets
print("🔹 Filtering good triplets...")
filtered_triplets = triplet_df[triplet_df["pos_score"] > triplet_df["neg_score"]]
filtered_triplets.to_csv("../data/final_triplets_for_training.csv", index=False)
print(f"✅ Kept {len(filtered_triplets)} triplets for training.")

# 6. Training the Dual Encoder
print("🔹 Starting model training...")
train_samples = []
for idx, row in filtered_triplets.iterrows():
    train_samples.append(
        InputExample(
            texts=[row["query"], row["positive_resume"], row["negative_resume"]],
            label=1.0  # Dummy label for Trainer
        )
    )

train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=batch_size)
train_loss = losses.MarginMSELoss(model=retriever_model)  # No margin arg because older version compatibility

retriever_model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=epochs,
    warmup_steps=100,
    output_path=output_dir
)

print(f"✅ Full fine-tuned model saved to {output_dir}!")
