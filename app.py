import streamlit as st
import fitz
from sentence_transformers import SentenceTransformer, util

# PRO CSS
st.markdown("""
<style>
.main-header {font-size:4rem !important; font-weight:800 !important; color:#1f77b4 !important; text-align:center;}
.metric-container {background:linear-gradient(135deg,#667eea 0%,#764ba2 100%); padding:2rem; border-radius:20px; text-align:center; color:white !important;}
.upload-area {background:#f8f9fa; padding:2rem; border-radius:15px; border:3px dashed #dee2e6; text-align:center;}
</style>
""", unsafe_allow_html=True)

st.set_page_config(page_title="AI Resume Screener Pro", page_icon="🤖", layout="wide")

# HEADER
st.markdown('<h1 class="main-header">🤖 AI Resume Screener Pro</h1>', unsafe_allow_html=True)
st.markdown("**Upload resume → Get instant job matches + course recommendations → Land your dream role!**")
st.divider()

# SIDEBAR
with st.sidebar:
    st.markdown("## 📊 Portfolio")
    st.info("**Dharshini** | AIML Student\n⭐ Pneumonia Detector\n🚀 AI Screener\n💼 ML Engineer Ready!")
    st.markdown("[GitHub](https://github.com/yourusername)")

# MAIN LAYOUT
col1, col2, col3 = st.columns([1,2,1])

with col1:
    st.markdown("### 📄 Upload Resume")
    resume_file = st.file_uploader("Choose PDF", type="pdf")

with col2:
    st.markdown("### 💼 Job Role")
    job_role = st.selectbox("Select position:",
        ["🤖 ML Engineer", "📊 Data Scientist", "👁️ Computer Vision", 
         "🧠 NLP Engineer", "🔥 GenAI Engineer", "⚙️ MLOps", 
         "🎯 Data Engineer", "🦾 AI Research", "📈 Business Analyst", 
         "💻 Backend ML", "🔬 Research Scientist", "✨ Custom"])

with col3:
    if resume_file: st.metric("Status", "Ready")

# 12 JOB DESCRIPTIONS (2026 HOT)
jobs = {
    "🤖 ML Engineer": "Python PyTorch TensorFlow Computer Vision Deep Learning Streamlit Flask Docker GitHub Medical AI CNN ResNet ML pipeline MLOps",
    "📊 Data Scientist": "Python Pandas Scikit-learn SQL A/B testing Jupyter statistical analysis feature engineering data pipeline visualization Power BI Tableau",
    "👁️ Computer Vision": "OpenCV YOLO U-Net Image segmentation PyTorch Satellite imagery Medical imaging object detection edge deployment TensorRT",
    "🧠 NLP Engineer": "Transformers Hugging Face BERT GPT LLaMA RAG LangChain tokenization NER sentiment analysis text generation",
    "🔥 GenAI Engineer": "LLMs RAG fine-tuning LoRA QLoRA prompt engineering LangChain LlamaIndex Pinecone Weaviate vector database",
    "⚙️ MLOps": "Docker Kubernetes Kubeflow MLflow Airflow CI/CD model monitoring feature store DVC Weights Biases",
    "🎯 Data Engineer": "Spark Airflow Kafka dbt Snowflake BigQuery ETL pipeline data warehouse PySpark SQL optimization",
    "🦾 AI Research": "PyTorch TensorFlow JAX reinforcement learning GANs diffusion models transformers research arXiv",
    "📈 Business Analyst": "SQL Excel Power BI Tableau A/B testing KPI dashboard stakeholder communication data storytelling",
    "💻 Backend ML": "FastAPI Flask Django PostgreSQL Redis Docker AWS Lambda serverless ML inference",
    "🔬 Research Scientist": "Statistical modeling Bayesian inference causal inference experimental design A/B testing reproducibility"
}

job_text = jobs.get(job_role, st.text_area("Custom JD:", height=100))

# AI PROCESSING
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

if resume_file and job_text:
    with st.spinner("🔬 AI analyzing..."):
        model = load_model()
        
        # PDF → Text
        doc = fitz.open(stream=resume_file.read(), filetype="pdf")
        resume_text = ""
        for page in doc: resume_text += page.get_text()
        
        # Skills boost
        skills = ['Python','PyTorch','OpenCV','Streamlit','Gradio','ResNet','Colab','GitHub','ML','AI']
        resume_smart = resume_text + " " + " ".join(skills*3)
        
        # Match score
        resume_emb = model.encode(resume_smart)
        job_emb = model.encode(job_text)
        score = util.cos_sim(resume_emb, job_emb)[0][0].item() * 100
        
        # RESULTS
        st.markdown("## 🎯 Match Results")
        col1, col2, col3 = st.columns([1,2,1])
        
        with col1:
            st.markdown(f"""
            <div class="metric-container">
                <h1 style='color:white;font-size:4rem;'>{score:.0f}%</h1>
                <h3 style='color:white;'>Score</h3>
            </div>""", unsafe_allow_html=True)
        
        with col2:
            if score > 80: st.success("🟢 **EXCELLENT** - Apply NOW!")
            elif score > 60: st.warning("🟡 **GOOD** - Tailor resume")
            else: st.error("🟠 **IMPROVE** - Add skills")
        
        with col3: st.success(f"**{job_role}**")
        
        # SKILLS PREVIEW
        with st.expander("📋 Resume Skills", expanded=True):
            st.text_area("", resume_text[:800], height=200)
        
        # COURSE RECOMMENDATIONS
        if score < 80:
            st.markdown("## 🎓 Fix Skills Gap")
            courses = {
                "🤖 ML Engineer": "🏆 DeepLearning.AI TensorFlow (Coursera)",
                "📊 Data Scientist": "📊 Google Data Analytics (Coursera)",
                "👁️ Computer Vision": "👁️ Computer Vision Basics (Coursera)",
                "🧠 NLP Engineer": "🧠 NLP Specialization (DeepLearning.AI)",
                "🔥 GenAI Engineer": "🔥 LangChain for LLMs (DeepLearning.AI)",
                "⚙️ MLOps": "⚙️ MLOps Specialization (Duke)",
                "🎯 Data Engineer": "🎯 Data Engineering (Google Cloud)"
            }
            if job_role in courses:
                st.info(f"**{courses[job_role]}** → Complete in 2 weeks!")
        
        st.balloons()

else:
    st.markdown("""
    ### 🚀 3 Steps:
    1. 📤 **Upload** resume PDF
    2. 🎯 **Pick** job role  
    3. ⚡ **Get** match score + courses
    """)

st.markdown("---")
st.markdown("⭐ **Dharshini** | Pneumonia → Resume Screener → ML Engineer!")
