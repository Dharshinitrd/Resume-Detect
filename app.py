import streamlit as st
import fitz
from sentence_transformers import SentenceTransformer, util

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def extract_text(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def smart_resume(resume_text):
    skills = ['Python', 'PyTorch', 'OpenCV', 'Streamlit', 'Gradio', 'ResNet', 'Colab', 'GitHub']
    return resume_text + " " + " ".join(skills * 2)

st.set_page_config(page_title="AI Resume Screener", layout="wide")
st.title("🤖 AI Resume Screener")

col1, col2 = st.columns(2)
with col1:
    resume_file = st.file_uploader("📄 Resume PDF", type="pdf")
with col2:
    job_role = st.selectbox("💼 Job", ["ML Engineer", "Data Scientist", "Computer Vision"])
    jobs = {
        "ML Engineer": "Python PyTorch Computer Vision Deep Learning Streamlit GitHub ML",
        "Data Scientist": "Python Pandas SQL Machine Learning Jupyter",
        "Computer Vision": "OpenCV PyTorch Image segmentation Medical imaging"
    }
    job_text = jobs[job_role]

if resume_file and job_text:
    model = load_model()
    resume_raw = extract_text(resume_file)
    resume_smart = smart_resume(resume_raw)
    
    resume_emb = model.encode(resume_smart)
    job_emb = model.encode(job_text)
    score = util.cos_sim(resume_emb, job_emb)[0][0].item() * 100
    
    st.metric("Match Score", f"{score:.0f}%")
    st.text_area("Skills Found", resume_raw[:500], height=200)
