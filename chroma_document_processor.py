import os
from langchain_community.document_loaders import PyPDFLoader, CSVLoader, JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.schema import Document


def load_document(file_path):
    """
    Load a document based on its file type (PDF, CSV, or JSON).
    """
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".csv"):
        loader = CSVLoader(file_path)
    elif file_path.endswith(".json"):
        loader = JSONLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path.split('.')[-1]}")
    return loader.load()


def process_and_query_document(file_path, question, openai_api_key, chroma_persist_directory="chroma_db"):
    """
    Process a document, split it into chunks, store it in Chroma, and answer a question using the ChatOpenAI model.
    
    Parameters:
    - file_path: str, path to the document file
    - question: str, the question to ask
    - openai_api_key: str, the OpenAI API key for authentication
    - chroma_persist_directory: str, directory to persist Chroma vector store data
    """
    # Set OpenAI API key as environment variable
    os.environ["OPENAI_API_KEY"] = openai_api_key

    # Load document
    try:
        documents = load_document(file_path)
        print(f"Loaded {len(documents)} document(s) from {file_path}.")
        print(f"Document preview: {documents[0].page_content[:500]}")
    except Exception as e:
        print(f"Error loading document: {e}")
        return

    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    print(f"Split documents into {len(chunks)} chunks.")

    # Initialize embeddings and vector store
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma(persist_directory=chroma_persist_directory, embedding_function=embeddings)
    
    # Add document chunks to Chroma
    vector_store.add_documents(chunks)
    print(f"Added documents to Chroma vector store at '{chroma_persist_directory}'.")

    # Use ChatOpenAI model for chat-based interaction
    chat_model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Set up RetrievalQA chain for question answering
    retriever = vector_store.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        llm=chat_model,
        chain_type="stuff",
        retriever=retriever
    )

    # Query the document using the question
    try:
        answer = qa_chain.invoke(question)
        print(f"Question: {question}\nAnswer: {answer}")

        # Ensure answer is a string before creating a Document
        if not isinstance(answer, str):
            answer = str(answer)

        # Store the LLM response in Chroma with metadata
        response_document = Document(page_content=answer, metadata={"source": "llm_response"})
        vector_store.add_documents([response_document])
        print("Stored LLM response in Chroma vector store.")

        # Verify the stored response using similarity search
        stored_response = vector_store.similarity_search("llm_response")
        print(f"Retrieved documents from Chroma vector store:")
        for doc in stored_response:
            print(f"Metadata: {doc.metadata}\nContent: {doc.page_content}\n")

    except Exception as e:
        print(f"Error during chat: {e}")


if __name__ == "__main__":
    # Example usage
    file_path = "your_document.pdf"  # Replace with your actual document path
    question = "What is the document about?"
    openai_api_key = "your-openai-api-key"  # Replace with your actual OpenAI API key
    chroma_persist_directory = "chroma_db"  # Directory to store Chroma database

    process_and_query_document(file_path, question, openai_api_key, chroma_persist_directory)
