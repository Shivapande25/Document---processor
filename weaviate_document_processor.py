import os
import weaviate
from langchain_community.document_loaders import PyPDFLoader, CSVLoader, JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import Document

def process_and_query_document(file_path, question, openai_api_key, weaviate_url):
    # Set the OpenAI API key as an environment variable
    os.environ["OPENAI_API_KEY"] = openai_api_key

    def load_document(file_path):
        """Load a document based on its file type."""
        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith(".csv"):
            loader = CSVLoader(file_path)
        elif file_path.endswith(".json"):
            loader = JSONLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path.split('.')[-1]}")
        return loader.load()

    try:
        documents = load_document(file_path)
        print(f"Loaded {len(documents)} document(s) from {file_path}.")
        print(f"Document preview: {documents[0].page_content[:500]}")
    except Exception as e:
        print(f"Error loading document: {e}")
        return

    # Split documents into chunks for processing
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    print(f"Split documents into {len(chunks)} chunks.")

    # Initialize OpenAI embeddings and Weaviate client
    embeddings = OpenAIEmbeddings()
    weaviate_client = weaviate.Client(url=weaviate_url)

    # Create schema if it does not already exist
    schema = {
        "classes": [
            {
                "class": "Document_index",
                "properties": [
                    {
                        "name": "content",
                        "dataType": ["text"]
                    }
                ]
            }
        ]
    }

    try:
        existing_schema = weaviate_client.schema.get()
        if not any(cls["class"] == "Document_index" for cls in existing_schema["classes"]):
            weaviate_client.schema.create(schema)
            print("Created Weaviate schema.")
        else:
            print("Weaviate schema already exists.")
    except Exception as e:
        print(f"Error creating schema: {e}")
        return

    # Add chunks to Weaviate
    batch = weaviate_client.batch
    batch.configure(batch_size=100)
    for chunk in chunks:
        batch.add_data_object(
            {"content": chunk.page_content},
            class_name="Document_index"
        )
    batch.create_objects()
    print(f"Added {len(chunks)} chunks to Weaviate.")

    # Set up RetrievalQA chain with ChatOpenAI
    chat_model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    # Define a function for searching Weaviate
    def search_weaviate(query):
        result = weaviate_client.query.get(class_name="Document_index", properties=["content"]).with_near_text({
            "concepts": [query]
        }).do()
        return result.get("data", {}).get("Get", {}).get("Document_index", [])

    try:
        # Get the answer using the ChatOpenAI model
        answer = chat_model(question)
        print(f"Question: {question}\nAnswer: {answer}")

        # Ensure the answer is a string before creating a Document
        if not isinstance(answer, str):
            answer = str(answer)

        # Store the LLM response in Weaviate
        response_document = Document(page_content=answer, metadata={"source": "llm_response"})
        batch = weaviate_client.batch
        batch.configure(batch_size=100)
        batch.add_data_object(
            {"content": response_document.page_content},
            class_name="Document_index"
        )
        batch.create_objects()
        print("Stored LLM response in Weaviate.")

        # Verify the stored response using similarity search
        stored_response = search_weaviate("llm_response")
        print("Retrieved documents from Weaviate:")
        for item in stored_response:
            print(f"Content: {item.get('content')}\n")

    except Exception as e:
        print(f"Error during chat: {e}")

# Example Usage
if __name__ == "__main__":
    # Replace these variables with your actual values
    FILE_PATH = "your_document"
    QUESTION = "your_question"
    OPENAI_API_KEY = "your_openai_key"
    weaviate_url="weaviate_url"  # Update this with your Weaviate UR
    
    # Call the function
    process_and_query_document(FILE_PATH, QUESTION, OPENAI_API_KEY, weaviate_url)
