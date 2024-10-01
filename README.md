# Document---processor
A repository for document processing and querying with Chroma and Weaviate.
# Document Processing and Querying with Chroma and Weaviate

This repository contains two scripts for processing documents (PDF, CSV, JSON) and querying them using two different vector store backends: **Chroma** and **Weaviate**. Both scripts utilize OpenAI's language models to generate responses based on document content.

## Table of Contents
- [Chroma Document Processor](#chroma-document-processor)
  - [Overview](#overview)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Example](#example)
- [Weaviate Document Processor](#weaviate-document-processor)
  - [Overview](#overview)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Example](#example)

---

## Chroma Document Processor

### Overview
The `chroma_document_processor.py` script processes documents in PDF, CSV, or JSON format, splits them into chunks, stores them in the **Chroma** vector store, and uses **OpenAI's GPT model** to answer questions based on the document content. It also stores the LLM responses in Chroma for future retrieval.

### Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/document-processors.git
   cd document-processors
2.Install the required dependencies:
pip install -r requirements.txt

3.Create an environment variable for your OpenAI API key:
export OPENAI_API_KEY="your-openai-api-key"


4.Usage:
a)Place the document (PDF, CSV, JSON) you want to process in the project directory.
b)Open the chroma_document_processor.py script and edit the following lines to match your file path and question:
file_path = "your_document.pdf"  # Replace with your actual document path
question = "What is the document about?"

5.Run the script:
python chroma_document_processor.py

6.Example:
If you have a PDF document named report.pdf and want to ask, "What is the main topic of the report?", modify the file_path and question variables and run the script.
The script will:
1)Load and split the document into chunks.
2)Store the document and its embeddings in the Chroma vector store.
3)Use OpenAI's GPT model to answer your question.
4)Store and verify the LLM response.





## Weaviate Document Processor
###Overview
The weaviate_document_processor.py script is similar to the Chroma processor but uses Weaviate, a cloud-native vector database, to store document embeddings and query responses. It allows for scalable and flexible querying with support for multi-modal data.

### Installation
1.Ensure that you have Weaviate running, either locally (e.g., via Docker) or via Weaviate Cloud:

2.To run Weaviate locally using Docker, use the following command:
docker run -d -p 8080:8080 semitechnologies/weaviate:latest

3.Install the necessary Python dependencies:
pip install -r requirements.txt

4.Export your OpenAI API key and set up the Weaviate endpoint:
export OPENAI_API_KEY="your-openai-api-key"
export WEAVIATE_ENDPOINT="http://localhost:8080"  # Update this if using Weaviate Cloud

5.Usage:
 1)Place your document (PDF, CSV, or JSON) in the project directory.
 2)Open the weaviate_document_processor.py script and edit the file_path and question variables:
  file_path = "your_document.pdf"  # Replace with your actual document path
   question = "What is the main focus of this document?"
   Run the script:
   python weaviate_document_processor.py
6.Example:
If you're using a CSV file named financial_data.csv, and you want to know, "What are the key trends in this financial data?", update the file path and question variables. Running the script will:
1)Load and split the document into chunks.
2)Store the document embeddings in Weaviate.
3)Query the document using GPT-based language models.
4)Store the LLM responses in Weaviate and verify the results.
### General Notes
Chroma: This vector store is lightweight and well-suited for local storage of document embeddings. It’s easy to set up and persist embeddings on disk.
Weaviate: This vector store is more scalable and is designed for enterprise-grade solutions with robust features like cloud-based storage, multi-modal data support, and customizable schemas.
You can choose either script depending on your project's needs. Both scripts allow easy document querying and storage using the OpenAI API.

### Dependencies
Python 3.8+
langchain_community
langchain_openai
langchain_chroma
langchain_weaviate
OpenAI API key
### To-Do
Extend the support for more document formats.
Add additional configurations for document processing.
Optimize the chunking strategy for larger documents.
