import streamlit as st
import requests
import json
from typing import List
import pandas as pd

# Page config
st.set_page_config(
    page_title="Legal Document Inconsistency Detector",
    page_icon="⚖️",
    layout="wide"
)

# API endpoint
API_URL = "http://localhost:8000/analyze"

def main():
    st.title("⚖️ Legal Document Inconsistency Detection System")
    st.markdown("Detect contradictions and inconsistencies in employment contracts")
    
    # Sidebar
    with st.sidebar:
        st.header("📋 Instructions")
        st.markdown("""
        1. Upload 2 or more legal documents
        2. Supported formats: TXT, PDF
        3. Click 'Analyze Documents'
        4. Review highlighted inconsistencies
        """)
        
        st.header("⚙️ Settings")
        similarity_threshold = st.slider(
            "Similarity Threshold", 
            min_value=0.5, 
            max_value=0.9, 
            value=0.65,
            help="Lower values detect more potential matches"
        )
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_files = st.file_uploader(
            "Choose documents",
            type=['txt', 'pdf'],
            accept_multiple_files=True,
            help="Upload employment contracts in TXT or PDF format"
        )
    
    with col2:
        if uploaded_files and len(uploaded_files) >= 2:
            if st.button("🔍 Analyze Documents", type="primary"):
                with st.spinner("Analyzing documents for inconsistencies..."):
                    analyze_documents(uploaded_files, similarity_threshold)
        else:
            st.info("Please upload at least 2 documents to analyze")
    
    # Display results if available
    if 'analysis_results' in st.session_state:
        display_results(st.session_state.analysis_results)

def analyze_documents(files: List, threshold: float):
    """Send documents to backend API and display results"""
    try:
        # Prepare files for upload
        files_to_upload = [("files", (file.name, file.getvalue(), file.type)) for file in files]
        
        # Make API request
        response = requests.post(
            API_URL,
            files=files_to_upload
        )
        
        if response.status_code == 200:
            st.session_state.analysis_results = response.json()
            st.success("Analysis complete!")
            st.rerun()
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
    
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to backend. Please ensure the FastAPI server is running.")
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")

def display_results(results: dict):
    """Display analysis results in a formatted way"""
    st.header("📊 Analysis Results")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Clauses Extracted", results['total_clauses_extracted'])
    with col2:
        st.metric("Comparisons Made", results['comparisons_made'])
    with col3:
        inconsistencies = results['summary']['inconsistencies_found']
        st.metric("⚠️ Inconsistencies Found", inconsistencies, delta_color="inverse")
    with col4:
        st.metric("Confidence Score", f"{results['summary']['avg_confidence']:.2%}")
    
    st.divider()
    
    # Display results in tabs
    tab1, tab2 = st.tabs(["🚨 Inconsistencies (Red Flags)", "✅ Consistent Clauses (Green Flags)"])
    
    with tab1:
        inconsistencies = [r for r in results['results'] if r['is_inconsistent']]
        if inconsistencies:
            for inc in inconsistencies:
                with st.expander(f"⚠️ {inc['inconsistency_type']} - Confidence: {inc['confidence']:.2%}", expanded=True):
                    colA, colB = st.columns(2)
                    
                    with colA:
                        st.markdown(f"**📄 {inc['clause_1']['document_name']}**")
                        st.markdown(f"*Clause {inc['clause_1']['clause_number']}*")
                        st.markdown(f"> {inc['clause_1']['text'][:300]}...")
                    
                    with colB:
                        st.markdown(f"**📄 {inc['clause_2']['document_name']}**")
                        st.markdown(f"*Clause {inc['clause_2']['clause_number']}*")
                        st.markdown(f"> {inc['clause_2']['text'][:300]}...")
                    
                    st.markdown(f"**🔍 Explanation:** {inc['explanation']}")
                    st.markdown(f"**Similarity Score:** {inc['similarity_score']:.2%}")
                    st.markdown("---")
        else:
            st.success("🎉 No inconsistencies detected!")
    
    with tab2:
        # Only show clauses that are truly identical
        consistent = [
            r for r in results['results']
            if not r['is_inconsistent']
            and r['clause_1']['text'].strip().lower() == r['clause_2']['text'].strip().lower()
        ]

        if consistent:
            st.markdown("### Consistent Clauses (Green Flags)")
            for cons in consistent[:20]:
                # Limit to first 20
                st.markdown(f"📄 **{cons['clause_1']['document_name']} Clause {cons['clause_1']['clause_number']}**")
                st.write(cons['clause_1']['text'])
                st.markdown(f"📄 **{cons['clause_2']['document_name']} Clause {cons['clause_2']['clause_number']}**")
                st.write(cons['clause_2']['text'])
                st.markdown(f"Similarity Score: {cons['similarity_score']:.2%}, Confidence: {cons['confidence']:.2%}")
                st.markdown("---")
                
        else:
            st.info("No exact consistent clauses found")

if __name__ == "__main__":
    main()