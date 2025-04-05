import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
import json
from langchain_groq import ChatGroq


# HTML and CSS for UI

# css = '''
# <style>
# .chat-message {
#     display: flex;
#     padding: 1.5rem;
#     border-radius: 0.5rem;
#     margin-bottom: 1rem;
#     align-items: center;
# }

# .chat-message.user {
#     background-color: #1f2833;
#     color: #66fcf1;
# }

# .chat-message.bot {
#     background-color: #0b0c10;
#     color: #45a29e;
# }

# .chat-message .avatar {
#     flex-shrink: 0;
#     width: 60px;
#     height: 60px;
#     margin-right: 1rem;
# }

# .chat-message .avatar img {
#     width: 100%;
#     height: 100%;
#     border-radius: 50%;
#     object-fit: cover;
# }

# .chat-message .message {
#     flex-grow: 1;
#     padding: 1rem;
#     color: inherit;
#     line-height: 1.5;
#     font-size: 1rem;
#     background-color: rgba(102, 252, 241, 0.1);
#     border-radius: 0.5rem;
# }
# </style>
# '''

# bot_template = '''
# <div class="chat-message bot">
#     <div class="avatar">
#         <img src="https://i.ibb.co/cN0nmSj/Screenshot-2023-05-28-at-02-37-21.png" alt="Bot Avatar">
#     </div>
#     <div class="message">{{MSG}}</div>
# </div>
# '''

# user_template = '''
# <div class="chat-message user">
#     <div class="avatar">
#         <img src="https://i.ibb.co/Z11D5Vh/IMG-8059.jpg" alt="User Avatar">
#     </div>
#     <div class="message">{{MSG}}</div>
# </div>
# '''
css = '''
<style>
.chat-message {
    display: flex;
    padding: 1.5rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    align-items: center;
}

.chat-message.user {
    background-color: #1f2833;
    color: #66fcf1;
}

.chat-message.bot {
    background-color: #0b0c10;
    color: #45a29e;
}

.chat-message .avatar {
    flex-shrink: 0;
    width: 60px;
    height: 60px;
    margin-right: 1rem;
}

.chat-message .avatar img {
    width: 100%;
    height: 100%;
    border-radius: 50%;
    object-fit: cover;
}

.chat-message .message {
    flex-grow: 1;
    padding: 1rem;
    color: inherit;
    line-height: 1.5;
    font-size: 1rem;
    background-color: rgba(102, 252, 241, 0.1);
    border-radius: 0.5rem;
}
</style>
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://i.ibb.co/cN0nmSj/Screenshot-2023-05-28-at-02-37-21.png" alt="Bot Avatar">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://i.ibb.co/Z11D5Vh/IMG-8059.jpg" alt="User Avatar">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

footer_template = '''
<div class="footer" style="text-align: center; margin-top: 2rem; font-size: 0.9rem; color: #66fcf1;">
    <p>&copy; 2025 Website created by Abdessamed</p>
</div>
'''

html_content = '''
<div>
    <!-- Chat Messages -->
    {{chat_messages}}

    <!-- Footer -->
    {{footer_template}}
</div>
'''



# def get_pdf_json(pdf_docs):
#     """Extract text from uploaded PDF documents and return as JSON."""
#     pdf_data = {}
#     for pdf in pdf_docs:
#         try:
#             pdf_reader = PdfReader(pdf)
#             text = ""
#             for page in pdf_reader.pages:
#                 text += page.extract_text() or ""
#             pdf_data[pdf.name] = text
#         except Exception as e:
#             st.error(f"Error processing {pdf.name}: {e}")
#     return pdf_data

def get_pdf_json(pdf_docs, output_file="extracted_data.json"):
    """
    Extract text from uploaded PDF documents and return as JSON.
    Also saves the JSON data to a specified file.
    """
    pdf_data = {}
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
            pdf_data[pdf.name] = {"text": text}  # Ensure text is stored as a string in a dictionary
        except Exception as e:
            st.error(f"Error processing {pdf.name}: {e}")

    # Save the extracted data as a JSON file
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(pdf_data, f, indent=4, ensure_ascii=False)
        st.success(f"JSON saved successfully to {output_file}")
    except Exception as e:
        st.error(f"Error saving JSON: {e}")

    return pdf_data

def get_text_chunks_from_json(pdf_data):
    """Split text into manageable chunks for embedding from JSON data."""
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = []
    metadata = []
    for pdf_name, pdf_info in pdf_data.items():  # pdf_info now is a dictionary
        text = pdf_info["text"]  # Accessing the text correctly from the dictionary
        pdf_chunks = text_splitter.split_text(text)
        chunks.extend(pdf_chunks)
        metadata.extend([{"source": pdf_name, "chunk": i + 1} for i in range(len(pdf_chunks))])
    return chunks, metadata

def get_vectorstore(text_chunks, metadata, embedding_model):
    """Create a FAISS vectorstore with the specified embedding model."""
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    vectorstore = FAISS.from_texts(
        texts=text_chunks,
        embedding=embeddings,
        metadatas=metadata
    )
    return vectorstore

def get_conversation_chain(vectorstore, llm_model):
    """Initialize a conversational retrieval chain with the selected LLM."""
    # llm = ChatOpenAI(model=llm_model)
    llm = ChatGroq(model="llama-3.3-70b-specdec") 
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    prompt_template = PromptTemplate(
        input_variables=["question", "context"],
        template="Use the following context to answer the question. If the answer is in the text, provide the answer along with the source:\n\nContext:\n{context}\n\nQuestion:\n{question}\n\nAnswer with source:"
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt_template}
    )

    return conversation_chain

def handle_userinput(user_question):
    """Handle user input and display conversation history."""
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    # Initialize session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question and st.session_state.conversation:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Settings")
        embedding_model = st.selectbox(
            "Embedding Model",
            ["sentence-transformers/all-MiniLM-L6-v2"]
            # ["aubmindlab/bert-base-arabertv02"]
        )
        llm_model = st.selectbox(
            "Language Model",
            # ["gpt-3.5-turbo"]
            ["llama-3.3-70b-specdec","gpt-3.5-turbo"]
        )

        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here", accept_multiple_files=True)

        if st.button("Process"):
            with st.spinner("Processing..."):
                try:
                    # Extract text from PDFs
                    pdf_data = get_pdf_json(pdf_docs)


                    # Split text into chunks
                    text_chunks, metadata = get_text_chunks_from_json(pdf_data)

                    # Create vectorstore
                    vectorstore = get_vectorstore(text_chunks, metadata, embedding_model)

                    # Initialize conversation chain
                    st.session_state.conversation = get_conversation_chain(vectorstore, llm_model)

                    st.success("Documents processed successfully. You can now ask questions!")
                except Exception as e:
                    st.error(f"An error occurred: {e}")

if __name__ == '__main__':
    main()

# 77777777777777
