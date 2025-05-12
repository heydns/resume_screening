from sentence_transformers import SentenceTransformer, util
import torch

# === New Job Description ===
job_description = """We are hiring a Data Analyst to join our analytics team. The ideal candidate should be proficient in SQL, Excel, and Python, with experience in data visualization tools such as Power BI or Tableau. Familiarity with statistical analysis and A/B testing is a plus."""

# === New CVs ===
resumes = {
    "CV1": "Data analyst with 3 years of experience using SQL, Excel, and Tableau for sales and operations dashboards. Strong background in A/B testing and statistical reporting.",
    "CV2": "Financial analyst with advanced Excel skills and experience building dashboards in Power BI. Recently completed a course in Python and SQL.",
    "CV3": "Software developer experienced in JavaScript and React. Built internal tools and automation scripts. No formal data analysis background.",
    "CV4": "Marketing analyst with Google Analytics and Excel expertise. Familiar with interpreting campaign data but limited SQL knowledge.",
    "CV5": "Business intelligence engineer skilled in SQL and data modeling, regularly collaborates with analysts and uses Tableau for reporting.",
}

# === Load Your Fine-Tuned Model ===
print("🔍 Loading model...")
model = SentenceTransformer("../dual_encoder_full_model")

# === Encode ===
print("⚙️ Encoding...")
resume_ids = list(resumes.keys())
resume_texts = list(resumes.values())
resume_embeddings = model.encode(resume_texts, convert_to_tensor=True)
jd_embedding = model.encode(job_description, convert_to_tensor=True)

# === Compute Similarities & Rank ===
scores = util.cos_sim(jd_embedding, resume_embeddings)[0]
top_indices = torch.topk(scores, k=5).indices.tolist()
ranked_resumes = [(resume_ids[i], scores[i].item(), resumes[resume_ids[i]]) for i in top_indices]

# === Display Results ===
print("\nTop Matching CVs for the Job Description:\n")
for res_id, score, text in ranked_resumes:
    print(f"Score: {score:.4f}")
    print(f"Candidate CV: {text}\n")
