from sentence_transformers import SentenceTransformer, util
import torch
from scipy.stats import kendalltau, spearmanr
import numpy as np

# === Job Description ===
job_description = """We are looking for a Frontend Web Developer with 3+ years of experience in React.js, JavaScript, and modern CSS frameworks (e.g., Tailwind, Bootstrap). The ideal candidate should have a portfolio of responsive web applications, experience with RESTful APIs, and familiarity with version control systems like Git. Bonus for candidates who understand basic backend integration and deployment on platforms like Vercel or Netlify."""

# === Resumes ===
resumes = {
    "A": """Frontend Developer with 4 years of experience using React, Tailwind, TypeScript, and Git. Led development of several dashboards deployed via Vercel. Deep experience in RESTful API integration, reusable components, and accessibility. Strong grasp of UX patterns and testing.""",
    "B": """Frontend Developer with 3 years of experience building web applications using React, Bootstrap, and JavaScript. Integrated REST APIs and deployed apps on Netlify. Comfortable with Git and basic UX design.""",
    "C": """Operations Manager in logistics. Strong Excel and project management skills, but no programming or web development experience.""",
    "D": """Mobile App Developer with 5 years of experience building apps in Flutter and Swift. Strong UI sense but no experience in web development.""",
    "E": """Full-stack Developer with focus on backend, but has worked with Vue.js and plain JavaScript for admin panels. Familiar with CSS and HTML, but no production React projects."""
}

# === Human Rankings ===
human_rankings = [
    ["A", "B", "E", "D", "C"],
    ["B", "A", "D", "E", "C"],
    ["A", "B", "D", "E", "C"],
    ["B", "A", "D", "E", "C"],
    ["A", "B", "D", "E", "C"]
]

# === Load the model ===
print("🔍 Loading model...")
model = SentenceTransformer("../dual_encoder_full_model")

# === Model Inference ===
print("⚙️ Encoding...")
resume_ids = list(resumes.keys())
resume_texts = list(resumes.values())
resume_embeddings = model.encode(resume_texts, convert_to_tensor=True)
jd_embedding = model.encode(job_description, convert_to_tensor=True)

# === Ranking ===
scores = util.cos_sim(jd_embedding, resume_embeddings)[0]
top_indices = torch.topk(scores, k=5).indices.tolist()
model_ranking = [resume_ids[i] for i in top_indices]

print(f"\n📊 Model Ranking: {model_ranking}")

# === Compare with Human Rankings ===
kendall_scores, spearman_scores = [], []
model_rank_dict = {res: i for i, res in enumerate(model_ranking)}

for i, human in enumerate(human_rankings):
    human_rank_dict = {res: i for i, res in enumerate(human)}
    model_ranks = [model_rank_dict[res] for res in resume_ids]
    human_ranks = [human_rank_dict[res] for res in resume_ids]

    kendall, _ = kendalltau(human_ranks, model_ranks)
    spearman, _ = spearmanr(human_ranks, model_ranks)
    kendall_scores.append(kendall)
    spearman_scores.append(spearman)

# === Final Scores ===
print("\n✅ Human Agreement Metrics:")
print(f"Average Kendall Tau:   {np.mean(kendall_scores):.4f}")
print(f"Average Spearman Rho:  {np.mean(spearman_scores):.4f}")
