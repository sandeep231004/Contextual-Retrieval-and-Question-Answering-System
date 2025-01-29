# Project: Contextual Retrieval and Question Answering System

## Overview
This project is an AI-driven **document retrieval and question-answering system** that leverages **vector databases,** **Pinecone, sentence embeddings, and a T5-based model**  and **LLM-based query answering** to extract and generate meaningful responses from a large corpus of documents. The pipeline includes:
- **Preprocessing**: Cleaning and chunking documents.
- **Embedding Generation**: Converting text into vector embeddings.   
- **Vector Storage & Retrieval**: Storing and fetching relevant document chunks from a Pinecone database.
- **Answer Generation**: Using an LLM to provide detailed and contextual responses.

---
## Features
✅ **Efficient Document Chunking**: Breaking down large documents into manageable, semantically relevant chunks.  
✅ **Vector-Based Search**: Queries are converted into embeddings to fetch the most relevant results.  
✅ **LLM-Powered Answers**: Uses a fine-tuned transformer model (FLAN-T5) to generate meaningful answers.  
✅ **Scalability**: Built to handle large datasets and return accurate results efficiently.  

---
## Installation & Setup

1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use 'venv\\Scripts\\activate'
   pip install -r requirements.txt
   ```
2. Set up **Pinecone API Key** and **Hugging Face Model Credentials**:
   ```bash
   export PINECONE_API_KEY='your_pinecone_key'
   export HF_MODEL='google/flan-t5-large'
   ```

---
## Loading and Preprocessing Pipeline
The preprocessing step ensures that text data is **cleaned, split into meaningful chunks**, and embedded for storage.


### 1️⃣ **Document Loading**
- **Purpose**: Loads raw text/PDF files.
- **Code Implementation**:
```python
def read_pdf(file_path: str) -> List[Dict[str, str]]: 
    """
    Returns: List[Dict[str, str]]: A list of dictionaries with 'page' and 'content'.
    """
    documents = []
    
    try:
        # Open the PDF file
        with open(file_path, 'rb') as pdf_file:
            reader = PyPDF2.PdfReader(pdf_file)
            for page_number, page in enumerate(reader.pages):
                text = page.extract_text()
                
                # Skip empty pages
                if text.strip():
                    documents.append({"page": page_number + 1, "content": text})
    except Exception as e:
        print(f"Error reading PDF: {e}")
    
    return documents
  ```

### 2️⃣ **Preprocessing and Text Chunking**
- **Purpose**: Breaks down large documents into **fixed-sized overlapping segments and removes unnecessary whitespace, special characters, and stopwords.**.
- **Code Implementation**:
```python
# Preprocess the extracted documents by cleaning and splitting them into chunks.

def preprocess_documents(documents: List[Dict[str, str]]) -> List[Dict[str, str]]: 
    """
    Args:
        documents (List[Dict[str, str]]): List of extracted documents with page and content.
    
    Returns:
        List[Dict[str, str]]: Preprocessed documents.
    """
    processed_docs = []
    chunk_size = 800  # Adjust based on your needs
    
    for doc in documents:
        content = doc['content']
        
        # Clean up content (optional: add specific cleaning rules)
        content = content.replace("\n", " ").strip()
        content = content.replace('\uf0b7', '-')
        
        # Split content into chunks if it's too long
        for i in range(0, len(content), chunk_size):
            chunk = content[i:i + chunk_size]
            processed_docs.append({"page": doc['page'], "chunk": chunk})
    
    return processed_docs
  ```

---
## Embedding Generation
- **Purpose**: Converts each chunk into **vector representations** using a pre-trained embedding model.
- **Model Used**: `ollama_emb`
- **Code Implementation**:
  ```python
  def generate_embeddings(text_chunks):
      embeddings = [embedding_model.embed_query(chunk) for chunk in text_chunks]
      return embeddings
  ```

---
## Storing & Retrieving Chunks
### 1️⃣ **Storing Data in Pinecone**
- **Purpose**: Saves **text embeddings** in a vector database for efficient searching.
- **Code Implementation**:
```python
# Generate embeddings and store in Pinecone
for document in final_doc:
    vector_id = document["id"]
    chunk_content = document["chunk"]
    metadata = document.get("metadata", {})

    # Check if the vector already exists in Pinecone
    existing_vector = index.fetch(ids=[vector_id])

    # If the vector exists, skip upsert (no comparison with metadata)
    if vector_id in existing_vector.get("vectors", {}):
        continue  # Skip this document

    # Get embedding for the content
    embedding = ollama_emb.embed_query(chunk_content)  # Adjust based on your embedding model's API

    metadata["chunk"] = chunk_content
    # Upsert the document and its embedding into Pinecone
    index.upsert([(vector_id, embedding, metadata)])
  ```

### 2️⃣ **Retrieving Relevant Chunks**
- **Purpose**: Fetches the top-K most similar chunks based on a query.
- **Code Implementation**:
  ```python
  def retrieve_chunks(query, top_k=5):
      query_vector = embedding_model.embed_query(query)
      results = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
      return [match['metadata']['chunk'] for match in results['matches']]
  ```

---
## Answer Generation
- **Purpose**: Uses a transformer model (FLAN-T5) to generate **detailed, context-aware answers**.
- **Code Implementation**:
  ```python
  def generate_answer(query, retrieved_chunks):
      context = " ".join(retrieved_chunks)
      input_text = f"question: {query} context: {context}"
      inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, 
    max_length=512 )
      output = model.generate(inputs.input_ids, max_length=150, num_beams=5, early_stopping=True, length_penalty=1.0)
  
      answer = tokenizer.decode(output[0], skip_special_tokens=True)
      return answer
  ```

---
## Example Usage
```python
query = "What are the key highlights of Budget 2023?"
retrieved_chunks = retrieve_chunks(query, top_k=3)
answer = generate_answer(query, retrieved_chunks)
print("Generated Answer:", answer)
```

---
## Future Improvements
🔹 **Improve Chunking**: Optimize chunk size dynamically for better context or try to use other embedding models like OpenAI Embedding etc...  

🔹 **Enhance Answer Generation**: For contextual and efficient answer generation you can use models like GPT, Llama etc...

🔹 **Multi-Modal Support**: Extend to PDFs, images, and audio-based retrieval.  

---
## Contributors
- **Sandeep Dabbada** - [GitHub](https://github.com/sandeep231004)


