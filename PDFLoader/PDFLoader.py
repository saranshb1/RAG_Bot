
import gradio as gr
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

file_path = "F:/Work/RAG_Bot/example_pdf/NASDAQ_AMRK_2023.pdf"
# loader = PyPDFLoader(file_path)

# docs = loader.load()

# print(len(docs))
# print(type(docs))

# print(docs[0].page_content[0:100])
# print(docs[0].metadata)


# def docs_loader():
#     docs = loader.load()
    
#     print(len(docs))
#     print(type(docs))

#     # print(docs[0].page_content[0:100])
#     # print(docs[0].metadata)
    
#     return docs


def upload_file(file_obj):
    list_file_path = []
    for idx, file in enumerate(file_obj):
        file_path = file_obj.name
        list_file_path.append(file_path)
    # print(file_path)
    # initialize_database(file_path, progress)
    return list_file_path

# Load PDF document and create doc splits
def docs_loader(list_file_path, chunk_size, chunk_overlap):
    # Processing for one document only
    # loader = PyPDFLoader(file_path)
    # pages = loader.load()
    loaders = [PyPDFLoader(x) for x in list_file_path]
    pages = []
    for loader in loaders:
        pages.extend(loader.load())
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size = 600, chunk_overlap = 50)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size, 
        chunk_overlap = chunk_overlap)
    doc_splits = text_splitter.split_documents(pages)
    return doc_splits


# docs_loader()
