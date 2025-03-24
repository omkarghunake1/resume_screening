import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Custom CSS for styling
st.markdown(
    """
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stTextArea, .stFileUploader {
        border-radius: 10px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        padding: 10px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text() or ""  # Handle NoneType cases
    return text.strip()

# Function to rank resumes based on job description
def rank_resumes(job_description, resumes):
    documents = [job_description] + resumes
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()
    
    job_description_vector = vectors[0]
    resume_vectors = vectors[1:]
    
    cosine_similarities = cosine_similarity([job_description_vector], resume_vectors).flatten()
    return cosine_similarities

# Streamlit app title with styling
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>AI Resume Screening & Candidate Ranking</h1>", unsafe_allow_html=True)

# Layout for job description input
st.markdown("### 📝 Enter the Job Description")
job_description = st.text_area("Paste the job description here", height=150)

# Layout for resume upload
st.markdown("### 📂 Upload Resumes")
uploaded_files = st.file_uploader("Upload PDF resumes", type=["pdf"], accept_multiple_files=True)

# Display ranking results if inputs are available
if uploaded_files and job_description:
    st.markdown("## 📊 Ranking Resumes")
    
    resumes = []
    resume_names = []
    
    # Progress bar for loading
    progress = st.progress(0)
    
    for idx, file in enumerate(uploaded_files):
        text = extract_text_from_pdf(file)
        if text:  # Avoid empty PDFs
            resumes.append(text)
            resume_names.append(file.name)
        progress.progress((idx + 1) / len(uploaded_files))  # Update progress bar

    if resumes:
        # Rank resumes
        scores = rank_resumes(job_description, resumes)

        # Create DataFrame with results
        results = pd.DataFrame({"Resume": resume_names, "Score": scores})
        results = results.sort_values(by="Score", ascending=False)

        # Displaying ranked resumes with color-coded scores
        st.dataframe(results.style.background_gradient(cmap="Blues"))
        
        # Display top candidate
        top_candidate = results.iloc[0]
        st.success(f"🏆 **Top Candidate:** {top_candidate['Resume']} (Score: {top_candidate['Score']:.2f})")

    else:
        st.error("No text extracted from the uploaded PDFs. Please check the files.")
