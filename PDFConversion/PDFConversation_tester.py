

import getpass
import os
import sys

working_directory = os.getcwd()
print(working_directory)

sys.path.append('F:/Work/RAG_Bot/PDFLoader')
#sys.path.insert()

from PDFLoader import docs_loader

doc_final = docs_loader()


#from langchain_openai import ChatOpenAI

#llm = ChatOpenAI(model="gpt-4o-mini")


from langchain_huggingface import HuggingFacePipeline
from langchain_chroma import Chroma
# from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain.vectorstores import FAISS

model_name = "tiiuae/falcon-7b-instruct"
embeddings = HuggingFaceEmbeddings(
    model_name=model_name
)


llm = HuggingFacePipeline.from_model_id(
    model_id= "tiiuae/falcon-7b-instruct",
    task="text2text-generation",
    huggingfacehub_api_token="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",
)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(doc_final)
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

retriever = vectorstore.as_retriever()


from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate



system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)


question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

results = rag_chain.invoke({"input": "What was AMRK's revenue in 2023?"})

print(results)


