# Project: AI-Powered Information Retrieval System

## Overview
This project is an AI-driven **information retrieval system** that leverages **vector databases** and **LLM-based query answering** to extract and generate meaningful responses from a large corpus of documents. The pipeline includes:
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

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/yourproject.git
   cd yourproject
   ```
2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use 'venv\\Scripts\\activate'
   pip install -r requirements.txt
   ```
3. Set up **Pinecone API Key** and **Hugging Face Model Credentials**:
   ```bash
   export PINECONE_API_KEY='your_pinecone_key'
   export HF_MODEL='google/flan-t5-large'
   ```

---
## Preprocessing Pipeline
The preprocessing step ensures that text data is **cleaned, split into meaningful chunks**, and embedded for storage.

### 1️⃣ **Document Loading & Cleaning**
- **Purpose**: Loads raw text/PDF files and removes unnecessary whitespace, special characters, and stopwords.
- **Code Implementation**:
  ```python
  def clean_text(text):
      text = text.replace('\n', ' ').strip()
      text = re.sub(r'[^a-zA-Z0-9.,!?\s]', '', text)
      return text.lower()
  ```

### 2️⃣ **Text Chunking**
- **Purpose**: Breaks down large documents into **fixed-sized overlapping segments**.
- **Code Implementation**:
  ```python
  def chunk_text(text, chunk_size=512, overlap=50):
      words = text.split()
      chunks = []
      for i in range(0, len(words), chunk_size - overlap):
          chunks.append(" ".join(words[i:i + chunk_size]))
      return chunks
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
  def store_chunks_in_pinecone(chunks, embeddings):
      for i, (chunk, vector) in enumerate(zip(chunks, embeddings)):
          index.upsert(vectors=[(str(i), vector, {'chunk': chunk})])
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
      input_ids = tokenizer(input_text, return_tensors="pt").input_ids
      output = model.generate(input_ids, max_length=500, num_beams=5, early_stopping=True)
      return tokenizer.decode(output[0], skip_special_tokens=True)
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
🔹 **Improve Chunking**: Optimize chunk size dynamically for better context.  
🔹 **Enhance Answer Generation**: Fine-tune FLAN-T5 for more structured answers.  
🔹 **Multi-Modal Support**: Extend to PDFs, images, and audio-based retrieval.  

---
## Contributors
- **Your Name** - [GitHub](https://github.com/yourusername)

---
## License
MIT License

