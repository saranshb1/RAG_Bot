

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings 
import chromadb
from pathlib import Path
import re
from unidecode import unidecode


# Create vector database
def create_db(splits, collection_name):
    embedding = HuggingFaceEmbeddings()
    new_client = chromadb.EphemeralClient()
    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embedding,
        client=new_client,
        collection_name=collection_name,
        # persist_directory=default_persist_directory
    )
    return vectordb


# Load vector database
def load_db():
    embedding = HuggingFaceEmbeddings()
    vectordb = Chroma(
        # persist_directory=default_persist_directory, 
        embedding_function=embedding)
    return vectordb


# Initialize database
def initialize_database(list_file_obj, chunk_size, chunk_overlap, progress=gr.Progress()):
    # Create list of documents (when valid)
    list_file_path = [x.name for x in list_file_obj if x is not None]
    # Create collection_name for vector database
    progress(0.1, desc="Creating collection name...")
    collection_name = create_collection_name(list_file_path[0])
    progress(0.25, desc="Loading document...")
    # Load document and create splits
    doc_splits = load_doc(list_file_path, chunk_size, chunk_overlap)
    # Create or load vector database
    progress(0.5, desc="Generating vector database...")
    # global vector_db
    vector_db = create_db(doc_splits, collection_name)
    progress(0.9, desc="Done!")
    return vector_db, collection_name, "Complete!"


# Generate collection name for vector database
#  - Use filepath as input, ensuring unicode text
def create_collection_name(filepath):
    # Extract filename without extension
    collection_name = Path(filepath).stem
    # Fix potential issues from naming convention
    ## Remove space
    collection_name = collection_name.replace(" ","-") 
    ## ASCII transliterations of Unicode text
    collection_name = unidecode(collection_name)
    ## Remove special characters
    #collection_name = re.findall("[\dA-Za-z]*", collection_name)[0]
    collection_name = re.sub('[^A-Za-z0-9]+', '-', collection_name)
    ## Limit length to 50 characters
    collection_name = collection_name[:50]
    ## Minimum length of 3 characters
    if len(collection_name) < 3:
        collection_name = collection_name + 'xyz'
    ## Enforce start and end as alphanumeric character
    if not collection_name[0].isalnum():
        collection_name = 'A' + collection_name[1:]
    if not collection_name[-1].isalnum():
        collection_name = collection_name[:-1] + 'Z'
    print('Filepath: ', filepath)
    print('Collection name: ', collection_name)
    return collection_name


