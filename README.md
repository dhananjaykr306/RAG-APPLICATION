# PDF Query and Summarization Application RAG (Retrival Augmented Generation)

This project is a **Streamlit application** designed to process PDF files, generate embeddings using **LangChain** and **Ollama Embeddings**, store the embeddings in **ChromaDB**, and allow users to query the uploaded PDF content. The app also provides summaries for the queried content using **Google Generative AI (Gemini model)**.

---

## Features

1. **Upload PDF**:
   - Upload a PDF file for content processing.
   - Extract and split the PDF content into manageable chunks using `RecursiveCharacterTextSplitter`.
   - Generate embeddings for each chunk using the `OllamaEmbeddings` model.
   - Store the embeddings in **ChromaDB** for future retrieval.

2. **Ask Query**:
   - Enter a query related to the uploaded PDF content.
   - Perform a similarity search in **ChromaDB** vector database to retrieve the most relevant content.
   - Use **Google Generative AI** to summarize the results based on the query.
   - Store the query and the corresponding summary back into **ChromaDB** for future reference.

3. **Cosine Similarity**:
   - Calculate the similarity between the query and document embeddings to identify the most relevant content.

4. **Help Section**:
   - Provides information about how to use the application and its functionalities.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/dhananjaykr306/RAG-APPLICATION.git
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   - Create a `.env` file in the project root.
   - Add your **Google Generative AI API key**:
     ```env
     GOOGLE_API_KEY=your_api_key_here
     ```

4. Run the application:
   ```bash
   streamlit run RAG_app.py
   ```

---

## Usage

### **Upload PDF**
- Navigate to the "Upload PDF" tab in the sidebar.
- Upload a PDF file.
- The app will:
  1. Extract and process the PDF content.
  2. Generate embeddings for the content chunks.
  3. Store the embeddings in **ChromaDB**.

### **Ask Query**
- Navigate to the "Ask Query" tab in the sidebar.
- Enter a query related to the uploaded PDF content.
- The app will:
  1. Generate an embedding for the query.
  2. Perform a similarity search in **ChromaDB**.
  3. Summarize the most relevant content using **Google Generative AI**.
  4. Display the summary and store it back in **ChromaDB**.

### **Help/Status**
- Use the "Help/Status" button to view information about the app and its functionality.

---

## File Structure

- `app.py`: Main application code.
- `requirements.txt`: List of Python dependencies.
- `.env`: Environment variables file (not included; you need to create it).

---

## Dependencies

- **Streamlit**: For building the web application interface.
- **PyPDF2**: For reading PDF files.
- **LangChain**: For document processing and embedding generation.
- **ChromaDB**: For storing and retrieving document embeddings.
- **Google Generative AI**: For generating query-based summaries.
- **dotenv**: For managing environment variables.
- **NumPy**: For mathematical operations, such as cosine similarity.

---

## Key Components

### **Cosine Similarity**
- Calculates similarity between the query embedding and document embeddings to identify the most relevant chunks.

### **ChromaDB Integration**
- Stores document and query embeddings for retrieval and similarity searches.

### **Generative AI**
- Summarizes the retrieved content based on the user query using **Google Gemini**.

---

## Requirements

- Python 3.8 or above.
- Google Generative AI API key.
- ChromaDB with persistence settings configured.

---



## Author
Dhananjay Kumar

For queries, reach out at: [dhananjaykr306@gmail.com](mailto:dhananjaykr306@gmail.com).

