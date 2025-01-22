import streamlit as st
from PyPDF2 import PdfReader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from chromadb import Client
from chromadb.api.types import Embedding
from dotenv import load_dotenv
import google.generativeai as genai
import os
import numpy as np
import time
from chromadb.config import Settings

# Function to calculate cosine similarity
def cosine_similarity(vec1, vec2):
    vec1, vec2 = np.array(vec1), np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


# Function to process query and generate summary
def summarization(content, chat_session, query):
    try:
        # Summarize the combined content
        summary_prompt = f"Give the answer of the {query} with respect to the {content} in points: {query} and {content}. Summarize in the form of a short answer. remove the irrelevent words"
        summary_response = chat_session.send_message(summary_prompt)
        summary = summary_response.text
        return summary

    except Exception as e:
        print(f"Error processing content: {e}")
        return "An error occurred while generating the summary."


# Main App
def main():
    # Initialize Chroma client with local persist directory
    persist_directory = r"C:\Users\dhana\Desktop\a\Project"
    chroma_client = Client(Settings(persist_directory=persist_directory))

    # App Mode Selection
    app_mode = st.sidebar.selectbox("Select Mode", ["Upload PDF", "Ask Query"])

    if app_mode == "Upload PDF":
        # PDF File Upload
        st.title("PDF File Uploader and Text Processing with LangChain")
        uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

        if uploaded_file is not None:
            # Save the uploaded file temporarily
            st.info("Generating PDF embedding...")
            with open("temp_uploaded_file.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Load the PDF using PyPDFLoader
            loader = PyPDFLoader("temp_uploaded_file.pdf")
            docs = loader.load()

            # Use RecursiveCharacterTextSplitter to split the document
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            final_docs = splitter.split_documents(docs)
            print(len(final_docs))

            # Create or load a collection in ChromaDB
            collection = chroma_client.get_or_create_collection("pdf_embeddings")

            # Get the count of existing documents to avoid overwriting
            existing_count = len(collection.get()["ids"])

            # Insert the documents into ChromaDB
            embeddings = OllamaEmbeddings(model="gemma:2b")
            t1 = time.time()
            for i, doc in enumerate(final_docs):
                # Use a unique ID based on the existing count
                t2 = time.time()
                unique_id = f"chunk_{existing_count + i}"
                print(f"Document {unique_id} adding to ChromaDB.")
                embedding: Embedding = embeddings.embed_documents([doc.page_content])[0]
                collection.add(
                    ids=[unique_id],
                    documents=[doc.page_content],
                    metadatas=[{"chunk_id": existing_count + i}],
                    embeddings=[embedding],
                )
                t3 = time.time()
                print(f"time taken for adding Document {unique_id}:", t3 - t2)
                print(f"Document {unique_id} added to ChromaDB.")
            t4 = time.time()
            print(f"Inserted {len(final_docs)} documents into ChromaDB. Total documents: {existing_count + len(final_docs)}.")
            print(f"Total time taken for adding all documents to ChromaDB:", t4 - t1)
            st.success(f"Inserted {len(final_docs)} documents into ChromaDB. Total documents: {existing_count + len(final_docs)}.")


    elif app_mode == "Ask Query":
        # Query Tab
        st.title("Ask a Query Based on Uploaded PDF")
        query = st.text_input("Enter your query:")

        if query:
            # Generate embedding for the query
            t1 = time.time()
            st.success("Generating query embedding")
            embeddings = OllamaEmbeddings(model="gemma:2b")
            query_embedding: Embedding = embeddings.embed_documents([query])[0]

            # Get or create the collection
            collection = chroma_client.get_or_create_collection("pdf_embeddings")

            # Perform similarity search
            st.success("Performing similarity search")
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=3,  # Get top 3 similar results
            )

            # Combine the top results into a single string for summarization
            combined_content = " ".join([" ".join(doc) if isinstance(doc, list) else doc for doc in results["documents"]])

            # Load API key from environment variable and generate summary
            try:
                load_dotenv()
                api_key = os.getenv("GOOGLE_API_KEY")
                if not api_key:
                    raise ValueError("API key not found. Ensure it's set in the .env file.")
                genai.configure(api_key=api_key)

                # Create a generative model
                generation_config = {
                    "temperature": 0,
                    "top_p": 0.95,
                    "top_k": 40,
                    "max_output_tokens": 8192,
                }
                model = genai.GenerativeModel(model_name="gemini-1.5-flash", generation_config=generation_config)
                chat_session = model.start_chat()

                summary = summarization(combined_content, chat_session, query)
                st.subheader("Summary of Results:")
                st.write(summary)
                t2 = time.time()
                st.write(f"Time taken: {t2-t1:.2f} seconds")

                # Insert the query and summary into ChromaDB for future use
                embedding_summary: Embedding = embeddings.embed_documents([summary])[0]
                collection.add(
                    ids=[f"query_{query}"],
                    documents=[summary],
                    metadatas=[{"query": query}],
                    embeddings=[embedding_summary],
                )
                st.success("Query and summary inserted into ChromaDB.")

            except Exception as e:
                st.error(f"Error: {e}")
        

    # Help Button at the bottom of the page
    if st.button("Help/Status"):
        st.subheader("Getting Help or Status")
        st.write(""" 
            1. **Upload PDF**: Upload a PDF file, and the app will process its content and generate document embeddings for storage in ChromaDB.
            2. **Ask Query**: Enter a query based on the uploaded PDF, and the app will search the most relevant content in the database, summarize it, and store the query and summary back into ChromaDB.
            3. **Commands**:
               - "Upload PDF": Uploads a PDF for processing.
               - "Ask Query": Asks a query based on the processed PDF.
        """)

if __name__ == "__main__":
    main()