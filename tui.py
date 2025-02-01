import streamlit as st
import tempfile
import time
import os
from pathlib import Path
from main import Agent

def setup_page_config():
    """Configure initial page settings and styling"""
    st.set_page_config(
        page_title="RAG Document Assistant",
        page_icon="📚",
        layout="centered"
    )
    
    st.markdown("""
        <style>
        @keyframes gradient {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        
        .streaming-container {
            border-radius: 10px;
            padding: 20px;
            margin: 10px 0;
            background: linear-gradient(90deg, #f0f7ff, #ffffff, #f0f7ff);
            background-size: 200% 200%;
            animation: gradient 3s ease infinite;
            border: 1px solid #e1e4e8;
        }
        
        .typing-indicator {
            display: inline-block;
            padding: 4px 8px;
            background-color: #f1f3f5;
            border-radius: 10px;
            margin: 5px 0;
            font-size: 0.8em;
        }
        
        .highlight-text {
            background: linear-gradient(120deg, #ffd700 0%, #ffd700 100%);
            background-repeat: no-repeat;
            background-size: 100% 0.2em;
            background-position: 0 88%;
            transition: background-size 0.25s ease-in;
        }
        
        .fade-in {
            opacity: 0;
            animation: fadeIn 0.5s ease-in forwards;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        
        .progress-bar {
            height: 4px;
            background-color: #e9ecef;
            border-radius: 2px;
            margin: 10px 0;
            overflow: hidden;
        }
        
        .progress-bar-fill {
            height: 100%;
            background: linear-gradient(90deg, #4c8bf5, #45a6f5);
            transition: width 0.2s ease;
        }
        </style>
    """, unsafe_allow_html=True)

def create_pdf_uploader():
    """Create an enhanced file uploader with preview capability"""
    uploaded_file = st.file_uploader(
        "📄 Upload your PDF document",
        type=["pdf"],
        help="Upload a PDF file to analyze or query its contents"
    )
    
    if uploaded_file:
        file_details = {
            "Filename": uploaded_file.name,
            "File size": f"{uploaded_file.size / 1024:.2f} KB",
            "File type": uploaded_file.type
        }
        
        with st.expander("📋 File Details", expanded=True):
            for key, value in file_details.items():
                st.text(f"{key}: {value}")
        
        return uploaded_file
    return None

def stream_response(text):
    """Create a streaming text effect with progress tracking"""
    message = st.empty()
    progress = st.progress(0)
    
    full_text = ""
    total_length = len(text)
    
    for i, char in enumerate(text):
        full_text += char
        message.markdown(f"```{full_text}```")
        progress.progress((i + 1) / total_length)
        time.sleep(0.01)
    
    progress.empty()
    return full_text

def main():
    setup_page_config()
    
    st.title("📚 RAG Document Assistant")
    st.markdown("---")
    
    # Initialize session state
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    if 'task' not in st.session_state:
        st.session_state.task = None
    if 'result' not in st.session_state:
        st.session_state.result = None
    
    # File upload section
    uploaded_file = create_pdf_uploader()
    
    if uploaded_file:
        # Task selection using radio instead of buttons
        task = st.radio(
            "Select your task",
            options=["📝 Summarize Document", "🔍 Query with RAG"],
            horizontal=True,
            key="task_selector"
        )
        
        # Query input for RAG
        query = None
        if "Query" in task:
            query = st.text_area(
                "🤔 Enter your question about the document",
                height=100,
                placeholder="What would you like to know about the document?",
                help="Enter a specific question about the document content"
            )
        
        # Process button
        process_button = st.button(
            "🚀 Process Document",
            help="Click to process your request",
            use_container_width=True
        )
        
        if process_button:
            if "Query" in task and not query:
                st.error("⚠️ Please enter your question in the text area above")
            else:
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.read())
                        file_path = tmp.name
                    
                    agent = Agent(os.getenv("COHERE_API_KEY"))
                    
                    with st.spinner("🔄 Processing your request..."):
                        if "Summarize" in task:
                            st.markdown("### 📋 Document Summary")
                            result = agent.summarise(file_path)
                        else:
                            st.markdown("### 💡 Answer")
                            result = agent.rag(file_path, query)
                    
                    # Display results with streaming effect
                    st.session_state.result = stream_response(result)
                    
                    # Add copy button for results
                    if st.session_state.result:
                        st.button(
                            "📋 Copy to Clipboard",
                            help="Copy the result to your clipboard",
                            on_click=lambda: st.write("Result copied to clipboard!")
                        )
                
                except Exception as e:
                    st.error(f"⚠️ An error occurred: {str(e)}")
                finally:
                    # Cleanup temporary file
                    if 'file_path' in locals():
                        Path(file_path).unlink(missing_ok=True)
    
    # Documentation/Help section
    with st.expander("ℹ️ How to use this tool"):
        st.markdown("""
        1. **Upload your document**: Start by uploading a PDF file using the file uploader above
        2. **Choose your task**: Select either:
           - 📝 **Summarize Document**: Generate a comprehensive summary
           - 🔍 **Query with RAG**: Ask specific questions about the content
        3. **Enter your query**: If using RAG, type your question in the text area
        4. **Process**: Click the Process Document button to get your results
        """)

if __name__ == "__main__":
    main()