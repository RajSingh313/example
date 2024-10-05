import os
from PyPDF2 import PdfReader
import docx
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.chains.question_answering import load_qa_chain
from streamlit_mic_recorder import speech_to_text
from io import BytesIO
from gtts import gTTS
from dotenv import load_dotenv
import streamlit as st

# Set page config at the very top
st.set_page_config(page_title="Gemini PDF/Voice Chatbot", page_icon="🤖")

# Load environment variables
load_dotenv()
genai_api_key = os.getenv("GOOGLE_API_KEY")

# Sidebar Styling
st.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
        padding: 20px;
    }
    .sidebar .element-container {
        margin-bottom: 20px;
    }
    .sidebar .stButton button {
        background-color: #007bff;
        color: white;
        border-radius: 8px;
        padding: 10px 15px;
    }
    .sidebar .stButton button:hover {
        background-color: #0056b3;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# General Styling for light/dark mode
st.markdown(
    """
    <style>
    body {
        font-family: 'Arial', sans-serif;
    }
    .user-text {
        background-color: #007bff;
        color: white;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 5px;
        text-align: left;
        max-width: 70%;
    }
    .bot-text {
        background-color: #f1f1f1;
        color: #333;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 5px;
        text-align: left;
        max-width: 70%;
    }
    .bot-text.dark-mode {
        background-color: #2c2f33;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Function to read and extract text from PDF and DOCX files
def get_files_text(uploaded_files):
    text = ""
    for uploaded_file in uploaded_files:
        _, file_extension = os.path.splitext(uploaded_file.name)
        if file_extension == ".pdf":
            text += get_pdf_text(uploaded_file) + "\n"
        elif file_extension == ".docx":
            text += get_docx_text(uploaded_file) + "\n"
        else:
            text += "Unsupported file type."
    return text

# Extract text from PDF with encoding handling
def get_pdf_text(pdf):
    reader = PdfReader(pdf)
    text = ""
    for page in reader.pages:
        try:
            page_text = page.extract_text()
            if page_text:
                text += page_text.encode('utf-8', 'ignore').decode('utf-8')  # Ignore problematic characters
        except Exception as e:
            print(f"Error extracting text from page: {str(e)}")
    return text

# Extract text from DOCX with encoding handling
def get_docx_text(doc_file):
    doc = docx.Document(doc_file)
    text = ""
    for para in doc.paragraphs:
        try:
            para_text = para.text
            text += para_text.encode('utf-8', 'ignore').decode('utf-8') + "\n"
        except Exception as e:
            print(f"Error extracting text from paragraph: {str(e)}")
    return text

# Function to split the text into chunks
def get_text_chunks(text, chunk_size=10000, overlap=1000):
    # Clean text by ignoring invalid unicode characters
    clean_text = text.encode('utf-8', 'ignore').decode('utf-8')
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_text(clean_text)

# Function to create vector store using FAISS
def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    return vector_store

# Chain for handling user queries with Gemini AI
def get_conversation_chain(vectorstore, language='en'):
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=genai_api_key)

    prompt_template = ChatPromptTemplate.from_messages(
        [("system", f"Please respond to user queries in {language}."), 
         ("user", "Context: {context}\nUser Question: {question}")]
    )

    chain = load_qa_chain(
        llm=model, 
        chain_type="stuff", 
        prompt=prompt_template
    )
    return chain

# Function to convert text to speech without saving audio every time
def text_to_speech(response_text, language="en"):
    lang_code = {"English": "en", "Urdu": "ur"}  # Add any other languages here
    try:
        tts = gTTS(text=response_text, lang=lang_code[language])
        audio_file = BytesIO()  # Create an in-memory bytes buffer
        tts.write_to_fp(audio_file)  # Write the audio data to the buffer
        audio_file.seek(0)  # Reset the file pointer to the beginning
        return audio_file
    except KeyError:
        raise ValueError(f"Language '{language}' not supported. Use 'English' or 'Urdu'.")

# Function to handle user input and get the response
def handle_user_input(user_question, language="en"):
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # Initialize chat history if not already present

    # Load embeddings and vector store each time a question is asked
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    try:
        # Load FAISS index; ensure it's available
        vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        
        if not vector_store:
            raise ValueError("Vector store is not available. Please upload documents first.")

        docs = vector_store.similarity_search(user_question)

        if not docs:
            return {"output_text": "No relevant documents found. Please ask another question."}

        chain = get_conversation_chain(vector_store, language)

        # Prepare the context for the chain
        context = " ".join([doc.page_content for doc in docs])
        response = chain({"input_documents": docs, "question": user_question, "context": context}, return_only_outputs=True)

        # Update conversation history
        st.session_state.chat_history.insert(0, {"role": "user", "content": user_question, "audio": None})
        response_audio = text_to_speech(response["output_text"], language)
        st.session_state.chat_history.insert(1, {"role": "bot", "content": response["output_text"], "audio": response_audio})

        return response

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return {"output_text": "An error occurred while processing your request."}

# Merged voice to text function with language option
def voice_to_text(language="en"):
    return speech_to_text(language=language, use_container_width=True, key=f"STT_{language}")

# Main Streamlit App
def main():
    # Sidebar options for language and document upload
    language = st.sidebar.selectbox("Select output Language", ["English", "Urdu"])
    upload_option = st.sidebar.selectbox("Choose input Mode", ["Text Q/A", "Voice Q/A"])
    uploaded_files = st.sidebar.file_uploader("Upload your PDF or DOCX files", accept_multiple_files=True)

    # Handle document processing in a separate function
    if st.sidebar.button("Process Documents"):
        with st.spinner("Processing..."):
            try:
                file_text = get_files_text(uploaded_files)
                if file_text:
                    text_chunks = get_text_chunks(file_text)
                    vector_store = get_vector_store(text_chunks)
                    vector_store.save_local("faiss_index")
                    st.success("Documents Processed")
                else:
                    st.error("No text extracted from documents. Please check the file formats.")
            except Exception as e:
                st.error(f"An error occurred while processing documents: {str(e)}")

    # Button to clear chat history
    if st.sidebar.button("Clear Chat History"):
        st.session_state.chat_history = []

    # Main content
    st.title("IntelliDoc 🤖")

    user_question = None
    if upload_option == "Voice Q/A":
        user_question = voice_to_text(language="ur" if language == "Urdu" else "en")
    else:
        user_question = st.text_input("Enter your question")

    if user_question:
        response = handle_user_input(user_question, language)

        # Display conversation history
        for chat in st.session_state.chat_history:
            if chat["role"] == "user":
                st.markdown(f'<div class="user-text">{chat["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="bot-text">{chat["content"]}</div>', unsafe_allow_html=True)
                st.audio(chat["audio"])

if __name__ == "__main__":
    main()
