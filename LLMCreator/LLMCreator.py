

from langchain_community.llms import HuggingFaceEndpoint
# from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import gradio as gr
import os


def initialize_LLM(llm_option, llm_temperature, max_tokens, top_k, vector_db, progress=gr.Progress()):
    # print("llm_option",llm_option)
    llm_name = list_llm[llm_option]
    print("llm_name: ",llm_name)
    qa_chain = initialize_llmchain(llm_name, llm_temperature, max_tokens, top_k, vector_db, progress)
    return qa_chain, "Complete!"


# Initialize langchain LLM chain
def initialize_llmchain(llm_model, temperature, max_tokens, top_k, vector_db, progress=gr.Progress()):
    progress(0.1, desc="Initializing HF tokenizer...")
    # HuggingFacePipeline uses local model
    # Note: it will download model locally...
    # tokenizer=AutoTokenizer.from_pretrained(llm_model)
    # progress(0.5, desc="Initializing HF pipeline...")
    # pipeline=transformers.pipeline(
    #     "text-generation",
    #     model=llm_model,
    #     tokenizer=tokenizer,
    #     torch_dtype=torch.bfloat16,
    #     trust_remote_code=True,
    #     device_map="auto",
    #     # max_length=1024,
    #     max_new_tokens=max_tokens,
    #     do_sample=True,
    #     top_k=top_k,
    #     num_return_sequences=1,
    #     eos_token_id=tokenizer.eos_token_id
    #     )
    # llm = HuggingFacePipeline(pipeline=pipeline, model_kwargs={'temperature': temperature})
    
    # HuggingFaceHub uses HF inference endpoints
    progress(0.5, desc="Initializing HF Hub...")
    # Use of trust_remote_code as model_kwargs
    # Warning: langchain issue
    # URL: https://github.com/langchain-ai/langchain/issues/6080
    if llm_model == "mistralai/Mixtral-8x7B-Instruct-v0.1":
        llm = HuggingFaceEndpoint(
            repo_id=llm_model, 
            # model_kwargs={"temperature": temperature, "max_new_tokens": max_tokens, "top_k": top_k, "load_in_8bit": True}
            temperature = temperature,
            max_new_tokens = max_tokens,
            top_k = top_k,
            load_in_8bit = True,
        )
    # elif llm_model in ["HuggingFaceH4/zephyr-7b-gemma-v0.1","mosaicml/mpt-7b-instruct"]:
    #     raise gr.Error("LLM model is too large to be loaded automatically on free inference endpoint")
    #     llm = HuggingFaceEndpoint(
    #         repo_id=llm_model, 
    #         temperature = temperature,
    #         max_new_tokens = max_tokens,
    #         top_k = top_k,
    #     )
    elif llm_model == "microsoft/phi-2":
        # raise gr.Error("phi-2 model requires 'trust_remote_code=True', currently not supported by langchain HuggingFaceHub...")
        llm = HuggingFaceEndpoint(
            repo_id=llm_model, 
            # model_kwargs={"temperature": temperature, "max_new_tokens": max_tokens, "top_k": top_k, "trust_remote_code": True, "torch_dtype": "auto"}
            temperature = temperature,
            max_new_tokens = max_tokens,
            top_k = top_k,
            trust_remote_code = True,
            torch_dtype = "auto",
        )
    elif llm_model == "TinyLlama/TinyLlama-1.1B-Chat-v1.0":
        llm = HuggingFaceEndpoint(
            repo_id=llm_model, 
            # model_kwargs={"temperature": temperature, "max_new_tokens": 250, "top_k": top_k}
            temperature = temperature,
            max_new_tokens = 250,
            top_k = top_k,
        )
    # elif llm_model == "meta-llama/Llama-2-7b-chat-hf":
    #     raise gr.Error("Llama-2-7b-chat-hf model requires a Pro subscription...")
    #     llm = HuggingFaceEndpoint(
    #         repo_id=llm_model, 
    #         # model_kwargs={"temperature": temperature, "max_new_tokens": max_tokens, "top_k": top_k}
    #         temperature = temperature,
    #         max_new_tokens = max_tokens,
    #         top_k = top_k,
    #     )
    else:
        llm = HuggingFaceEndpoint(
            repo_id=llm_model, 
            # model_kwargs={"temperature": temperature, "max_new_tokens": max_tokens, "top_k": top_k, "trust_remote_code": True, "torch_dtype": "auto"}
            # model_kwargs={"temperature": temperature, "max_new_tokens": max_tokens, "top_k": top_k}
            temperature = temperature,
            max_new_tokens = max_tokens,
            top_k = top_k,
        )
    
    progress(0.75, desc="Defining buffer memory...")
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key='answer',
        return_messages=True
    )
    # retriever=vector_db.as_retriever(search_type="similarity", search_kwargs={'k': 3})
    retriever=vector_db.as_retriever()
    progress(0.8, desc="Defining retrieval chain...")
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=retriever,
        chain_type="stuff", 
        memory=memory,
        # combine_docs_chain_kwargs={"prompt": your_prompt})
        return_source_documents=True,
        #return_generated_question=False,
        verbose=False,
    )
    progress(0.9, desc="Done!")
    return qa_chain



