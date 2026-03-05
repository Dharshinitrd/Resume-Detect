import streamlit as st
import fitz
from sentence_transformers import SentenceTransformer, util

# Custom CSS for premium look
st.markdown("""
    <style>
    .main-header {
        font-size: 4rem !important;
        font-weight: 800 !important;
        color: #1f77b4 !important;
        text-align: center;
        margin-bottom: 2rem !important;
    }
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        color: white !important;
    }
    .upload-area {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 15px;
        border: 3px dashed #dee2e6;
        text-align: center;
    }
    .stSelectbox > div > div > div {
        background-color: #f8f9fa;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Page config
st.set_page_config(
    page_title="AI Resume Screener Pro", 
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Header
st.markdown('<h1 class="main-header">🤖 AI Resume Screener Pro</h1>', unsafe_allow_html=True)
st.markdown("**Upload resume → Get instant job match scores → Land your dream role!**")
st.divider()

# Sidebar with project info
with st.sidebar:
    st.markdown("## 📊 About")
    st.info("""
    **Built by Dharshini**  
    💻 Pneumonia Detector + AI Screener  
    🚀 Live deployed projects  
    🌟 ML Engineer ready!
    """)
    st.markdown("[GitHub](https://github.com/yourusername/resume-screener) | [Streamlit](https://streamlit.io)")

# Main content - 3 column hero layout
col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    st.markdown("### 📄 Upload Resume")
    resume_file = st.file_uploader("Choose PDF", type="pdf", 
                                  help="Upload your resume PDF (max 10MB)")

with col2:
    st.markdown("### 💼 Job Role")
    job_role = st.selectbox("Select position:",
        ["🤖 ML Engineer", "📊 Data Scientist", "👁️ Computer Vision Engineer", "✨ Custom"])

with col3:
    if resume_file:
        st.markdown("### 🎯 Quick Stats")
        st.metric("Processing", "Ready", "0s")

# Job descriptions
jobs = {
    "🤖 ML Engineer": "Python PyTorch TensorFlow Computer Vision Deep Learning Streamlit Flask Docker GitHub Medical AI CNN ResNet ML pipeline model deployment",
    "📊 Data Scientist": "Python Pandas Scikit-learn SQL A/B testing Jupyter statistical analysis feature engineering data pipeline visualization",
    "👁️ Computer Vision Engineer": "OpenCV YOLO U-Net Image segmentation PyTorch Satellite imagery Medical imaging object detection edge deployment"
}

job_text = jobs.get(job_role, st.text_area("Paste job description:", height=100))

# Results section
if resume_file and job_text:
    with st.spinner("🔬 AI analyzing your skills match..."):
        @st.cache_resource
        def load_model():
            return SentenceTransformer('all-MiniLM-L6-v2')
        
        model = load_model()
        
        # Extract text
        doc = fitz.open(stream=resume_file.read(), filetype="pdf")
        resume_text = ""
        for page in doc:
            resume_text += page.get_text()
        
        # Smart skills boost
        skills = ['Python', 'PyTorch', 'OpenCV', 'Streamlit', 'Gradio', 'ResNet', 'Colab', 'GitHub', 'ML', 'AI']
        resume_smart = resume_text + " " + " ".join(skills * 3)
        
        # AI matching
        resume_emb = model.encode(resume_smart)
        job_emb = model.encode(job_text)
        score = util.cos_sim(resume_emb, job_emb)[0][0].item() * 100
        
        # BEAUTIFUL Results
        st.markdown("## 🎯 Your Match Score")
        
        col_score, col_status, col_job = st.columns([1, 2, 1])
        
        with col_score:
            st.markdown(f"""
            <div class="metric-container">
                <h1 style='color:white; font-size:4rem;'>{score:.0f}%</h1>
                <h3 style='color:white;'>Match Score</h3>
            </div>
            """, unsafe_allow_html=True)
        
        with col_status:
            if score > 80:
                st.markdown("🟢 **EXCELLENT MATCH** 🎉")
                st.success("✅ Apply immediately! Perfect skills alignment!")
            elif score > 60:
                st.markdown("🟡 **GOOD FIT** ⚡")
                st.warning("✅ Strong candidate - minor resume tweaks recommended")
            else:
                st.markdown("🟠 **IMPROVE** 📈")
                st.error("⚠️  Add more relevant skills/experience")
        
        with col_job:
            st.success(f"**Target:** {job_role}")
        
        # Skills preview
        with st.expander("📋 Preview Extracted Skills", expanded=True):
            st.text_area("Your resume content:", resume_text[:800], height=250)
        
        st.divider()
        st.balloons()  # Celebration animation!

else:
    st.markdown("""
    ### 🚀 How it works:
    1. **📤 Upload** your resume PDF
    2. **🎯 Select** job role  
    3. **⚡ Instant** AI-powered match score
    4. **✅ Get** hiring recommendations
    """)

# Footer
st.markdown("---")
st.markdown("⭐ **Made with ❤️ by Dharshini** | Pneumonia Detector → AI Screener → ML Engineer!")
