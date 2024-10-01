
import gradio as gr
from LLMCreator import initialize_LLM
from PDFConversion.DBCreator import initialize_database


list_llm = ["mistralai/Mistral-7B-Instruct-v0.2", "mistralai/Mixtral-8x7B-Instruct-v0.1", "mistralai/Mistral-7B-Instruct-v0.1", \
    "google/gemma-7b-it","google/gemma-2b-it", \
    "HuggingFaceH4/zephyr-7b-beta", "HuggingFaceH4/zephyr-7b-gemma-v0.1", \
    "meta-llama/Llama-2-7b-chat-hf", "microsoft/phi-2", \
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "mosaicml/mpt-7b-instruct", "tiiuae/falcon-7b-instruct", \
    "google/flan-t5-xxl"
]

list_llm_simple = [os.path.basename(llm) for llm in list_llm]

def format_chat_history(message, chat_history):
    formatted_chat_history = []
    for user_message, bot_message in chat_history:
        formatted_chat_history.append(f"User: {user_message}")
        formatted_chat_history.append(f"Assistant: {bot_message}")
    return formatted_chat_history


def conversation(qa_chain, message, history):
    formatted_chat_history = format_chat_history(message, history)
    #print("formatted_chat_history",formatted_chat_history)
    
    # Generate response using QA chain
    response = qa_chain({"question": message, "chat_history": formatted_chat_history})
    response_answer = response["answer"]
    if response_answer.find("Helpful Answer:") != -1:
        response_answer = response_answer.split("Helpful Answer:")[-1]
    response_sources = response["source_documents"]
    response_source1 = response_sources[0].page_content.strip()
    response_source2 = response_sources[1].page_content.strip()
    response_source3 = response_sources[2].page_content.strip()
    # Langchain sources are zero-based
    response_source1_page = response_sources[0].metadata["page"] + 1
    response_source2_page = response_sources[1].metadata["page"] + 1
    response_source3_page = response_sources[2].metadata["page"] + 1
    # print ('chat response: ', response_answer)
    # print('DB source', response_sources)
    
    # Append user message and response to chat history
    new_history = history + [(message, response_answer)]
    # return gr.update(value=""), new_history, response_sources[0], response_sources[1] 
    return qa_chain, gr.update(value=""), new_history, response_source1, response_source1_page, response_source2, response_source2_page, response_source3, response_source3_page
    




def main_bot():
    with gr.Blocks(theme="base") as demo:
        vector_db = gr.State()
        qa_chain = gr.State()
        collection_name = gr.State()
        
        gr.Markdown(
        """<center><h2>PDF-based chatbot</center></h2>
        <h3>Ask any questions about your PDF documents</h3>""")
        gr.Markdown(
        """<b>Note:</b> This AI assistant, using Langchain and open-source LLMs, performs retrieval-augmented generation (RAG) from your PDF documents. \
        The user interface explicitly shows multiple steps to help understand the RAG workflow. 
        This chatbot takes past questions into account when generating answers (via conversational memory), and includes document references for clarity purposes.<br>
        <br><b>Warning:</b> This space uses the free CPU Basic hardware from Hugging Face. Some steps and LLM models used below (free inference endpoints) can take some time to generate a reply.
        """)
        
        with gr.Tab("Step 1 - Upload PDF"):
            with gr.Row():
                document = gr.Files(height=100, file_count="multiple", file_types=["pdf"], interactive=True, label="Upload your PDF documents (single or multiple)")
                # upload_btn = gr.UploadButton("Loading document...", height=100, file_count="multiple", file_types=["pdf"], scale=1)
        
        with gr.Tab("Step 2 - Process document"):
            with gr.Row():
                db_btn = gr.Radio(["ChromaDB"], label="Vector database type", value = "ChromaDB", type="index", info="Choose your vector database")
            with gr.Accordion("Advanced options - Document text splitter", open=False):
                with gr.Row():
                    slider_chunk_size = gr.Slider(minimum = 100, maximum = 1000, value=600, step=20, label="Chunk size", info="Chunk size", interactive=True)
                with gr.Row():
                    slider_chunk_overlap = gr.Slider(minimum = 10, maximum = 200, value=40, step=10, label="Chunk overlap", info="Chunk overlap", interactive=True)
            with gr.Row():
                db_progress = gr.Textbox(label="Vector database initialization", value="None")
            with gr.Row():
                db_btn = gr.Button("Generate vector database")
            
        with gr.Tab("Step 3 - Initialize QA chain"):
            with gr.Row():
                llm_btn = gr.Radio(list_llm_simple, \
                    label="LLM models", value = list_llm_simple[0], type="index", info="Choose your LLM model")
            with gr.Accordion("Advanced options - LLM model", open=False):
                with gr.Row():
                    slider_temperature = gr.Slider(minimum = 0.01, maximum = 1.0, value=0.7, step=0.1, label="Temperature", info="Model temperature", interactive=True)
                with gr.Row():
                    slider_maxtokens = gr.Slider(minimum = 224, maximum = 4096, value=1024, step=32, label="Max Tokens", info="Model max tokens", interactive=True)
                with gr.Row():
                    slider_topk = gr.Slider(minimum = 1, maximum = 10, value=3, step=1, label="top-k samples", info="Model top-k samples", interactive=True)
            with gr.Row():
                llm_progress = gr.Textbox(value="None",label="QA chain initialization")
            with gr.Row():
                qachain_btn = gr.Button("Initialize Question Answering chain")

        with gr.Tab("Step 4 - Chatbot"):
            chatbot = gr.Chatbot(height=300)
            with gr.Accordion("Advanced - Document references", open=False):
                with gr.Row():
                    doc_source1 = gr.Textbox(label="Reference 1", lines=2, container=True, scale=20)
                    source1_page = gr.Number(label="Page", scale=1)
                with gr.Row():
                    doc_source2 = gr.Textbox(label="Reference 2", lines=2, container=True, scale=20)
                    source2_page = gr.Number(label="Page", scale=1)
                with gr.Row():
                    doc_source3 = gr.Textbox(label="Reference 3", lines=2, container=True, scale=20)
                    source3_page = gr.Number(label="Page", scale=1)
            with gr.Row():
                msg = gr.Textbox(placeholder="Type message (e.g. 'What is this document about?')", container=True)
            with gr.Row():
                submit_btn = gr.Button("Submit message")
                clear_btn = gr.ClearButton([msg, chatbot], value="Clear conversation")
            
        # Preprocessing events
        #upload_btn.upload(upload_file, inputs=[upload_btn], outputs=[document])
        db_btn.click(initialize_database, \
            inputs=[document, slider_chunk_size, slider_chunk_overlap], \
            outputs=[vector_db, collection_name, db_progress])
        qachain_btn.click(initialize_LLM, \
            inputs=[llm_btn, slider_temperature, slider_maxtokens, slider_topk, vector_db], \
            outputs=[qa_chain, llm_progress]).then(lambda:[None,"",0,"",0,"",0], \
            inputs=None, \
            outputs=[chatbot, doc_source1, source1_page, doc_source2, source2_page, doc_source3, source3_page], \
            queue=False)

        # Chatbot events
        msg.submit(conversation, \
            inputs=[qa_chain, msg, chatbot], \
            outputs=[qa_chain, msg, chatbot, doc_source1, source1_page, doc_source2, source2_page, doc_source3, source3_page], \
            queue=False)
        submit_btn.click(conversation, \
            inputs=[qa_chain, msg, chatbot], \
            outputs=[qa_chain, msg, chatbot, doc_source1, source1_page, doc_source2, source2_page, doc_source3, source3_page], \
            queue=False)
        clear_btn.click(lambda:[None,"",0,"",0,"",0], \
            inputs=None, \
            outputs=[chatbot, doc_source1, source1_page, doc_source2, source2_page, doc_source3, source3_page], \
            queue=False)
    demo.queue().launch(debug=True)


