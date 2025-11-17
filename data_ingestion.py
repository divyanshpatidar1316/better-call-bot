import os
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# Set up environment variables
load_dotenv()

# Load and embed the documents
def embed_and_save_documents():
    # Initialize HuggingFace embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    loader = PyPDFDirectoryLoader("./legal_documents")
    print("Loader initialised")
    docs = loader.load()
    print(f"Loading the docs - Total documents loaded: {len(docs)}")
    
    if not docs:
        print("No documents found in the 'legal_documents' folder.")
        print("Please add your PDF files (e.g., 'the_constitution_of_india.pdf') to the 'legal_documents' folder and try again.")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_documents = text_splitter.split_documents(docs)
    print(f"Splitting the docs - Total chunks created: {len(final_documents)}")
    
    # Ensure metadata includes the source file name
    for doc in final_documents:
        if 'source' in doc.metadata:
            source_file = doc.metadata['source']
            doc.metadata['source'] = os.path.basename(source_file)
        else:
            # If source metadata is not present, add it
            doc.metadata['source'] = os.path.basename(loader.directory)
    
    # Ensure the payload size is within limits by batching the documents
    batch_size = 100  # Adjust batch size as needed
    batched_documents = [final_documents[i:i + batch_size] for i in range(0, len(final_documents), batch_size)]
    vector_stores = []
    
    print(f"Processing {len(batched_documents)} batches...")
    for idx, batch in enumerate(batched_documents):
        print(f"Processing batch {idx + 1}/{len(batched_documents)}...")
        vector_store = FAISS.from_documents(batch, embeddings)
        vector_stores.append(vector_store)
    print("Created batched documents")
    
    # Merge the vector stores
    print("Merging vector stores...")
    vectors = vector_stores[0]
    for vector_store in vector_stores[1:]:
        vectors.merge_from(vector_store)
    print("Merged the vectors")
    
    # Save the vector store to disk
    vectors.save_local("vector_db")
    print("Vectors saved successfully to 'vector_db' directory!")

if __name__ == "__main__":
    embed_and_save_documents()