# Import necessary libraries
import warnings
from langchain_core._api.deprecation import LangChainDeprecationWarning

warnings.filterwarnings("ignore", category=LangChainDeprecationWarning, module="llama_index")

import getpass
import os
import gradio as gr
from openai import OpenAI

# Set the environment

from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex, StorageContext
def create_multimodal_index(documents, images):
    multimodal_docs = []
    for doc, img_path in zip(documents, images):
        # Create a document with text and image embedding
        embedding = multimodal_embedding(doc.text, img_path)
        multimodal_doc = Document(text=doc.text, embedding=embedding)
        multimodal_docs.append(multimodal_doc)

from llama_index.llms.nvidia import NVIDIA
Settings.llm = NVIDIA(model="meta/llama-3.1-8b-instruct")

#import PyPDF2

#def extract_text_from_pdf(file_obj):
#    pdf_reader = PyPDF2.PdfReader(file_obj)
#    text = ""
#    for page in pdf_reader.pages:
#        text += page.extract_text()
#    return text


from llama_index.readers.file import (
    DocxReader,
    PDFReader,
    HTMLTagReader,
    ImageReader,
    PptxReader,
    CSVReader,
)

from llama_index.embeddings.nvidia import NVIDIAEmbedding
Settings.embed_model = NVIDIAEmbedding(model="NV-Embed-QA", truncate="END")

from llama_index.vector_stores.milvus import MilvusVectorStore

from llama_index.core.node_parser import SentenceSplitter
Settings.text_splitter = SentenceSplitter(chunk_size=400)

# Import libraries for image processing and URL scraping (if needed)
from PIL import Image  # For basic image processing
from bs4 import BeautifulSoup  # For basic URL scraping

# Import Nemo modules

from nemoguardrails import LLMRails, RailsConfig

# Define a RailsConfig object
config = RailsConfig.from_path("./Config")
rails = LLMRails(config)

# Initialize global variables for the index and query engine
index = None
query_engine = None

# Function to get file names from file objects
def get_files_from_input(file_objs):
    if not file_objs:
        return []
    return [file_obj.name for file_obj in file_objs]

# Function to load documents and create the index

# In your load_documents function:
def load_documents(file_objs, url=None):
    global index, query_engine
    all_texts = []
    all_images = []

    file_extractor = {
        ".pdf": PDFReader(),
        ".csv": CSVReader(),
        ".html": HTMLTagReader(),
        ".pptx": PptxReader(),
        ".docx": DocxReader(),
        ".jpeg": ImageReader(), 
        ".jpg": ImageReader(),
        ".png": ImageReader(),
        }
    file_results = []
    for file_obj in file_objs:
        file_extension = os.path.splitext(file_obj.name)[1].lower()
        if file_extension in file_extractor:
            extractor = file_extractor[file_extension]
            if file_extension in (".jpg", ".png", ".jpeg"):
                all_images.append(file_obj.name)
                file_results.append(f"Image processed: {file_obj.name}")
            else:
                try:
                    texts = extractor.extract(file_obj)
                    all_texts.extend(texts)
                    file_results.append(f"Text extracted from: {file_obj.name}")
                except Exception as e:
                    file_results.append(f"Error extracting text from {file_obj.name}: {e}")
        else:
            file_results.append(f"Unsupported file type ignored: {file_extension}")
    return "\n".join(file_results) 

    if url:
        try:
            all_texts.append(process_url(url))
        except Exception as e:
            print(f"Error processing URL: {e}")

    if not all_texts and not all_images:
        return f"No documents found in the selected files or URL."
            

    # Create a Milvus vector store and storage context
        # vector_store = MilvusVectorStore(
        #    host="127.0.0.1",
        #    port=19530,
        #    dim=1024,
        #    collection_name="your_collection_name",
        #    gpu_id=0,  # Specify the GPU ID to use
        #    output_fields=["field1","field2"]
        #)
        
    vector_store = MilvusVectorStore(uri="./milvus_demo.db", dim=1024, overwrite=True,output_fields=[])
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    #create the multimodal index
    multimodal_docs = create_multimodal_index(all_texts, all_images)
    index = VectorStoreIndex.from_documents(multimodal_docs, storage_context=storage_context)
        
    # Create the query engine after the index is created
    query_engine = index.as_query_engine()
    return f"Loaded {len(all_texts)} text segments and {len(all_images)} images."
    

# Function to extract text from image (replace with your preferred method)
from PIL import Image
import numpy as np

# Add to your existing ImageReader or create a new function for processing:
def process_image(file_path):
    with Image.open(file_path) as img:
        img_array = np.array(img)
        # Here you could do image processing like resizing, filtering, etc.
        # Example: 
        # img_resized = img.resize((300, 300))
        # Convert back to text or process further for embeddings
        # For now, let's just convert image to a string representation
        return str(img_array.tolist())


#URL processing
import requests
from bs4 import BeautifulSoup

def process_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        # Extract only relevant text, remove scripts and styles
        for script in soup(["script", "style"]):
            script.decompose()
        text = soup.get_text(separator=' ', strip=True)
        return text
    except requests.RequestException as e:
        return f"Error processing URL: {e}"
        

# Multimodal embedding
from sentence_transformers import SentenceTransformer

# Assuming you're using a model that can handle both text and images
model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')  # Example model

def multimodal_embedding(text, img_path=None):
    if img_path:
        # Process image, convert to embedding - this is a placeholder
        # You'll need a model like CLIP for actual image-text embeddings
        img_emb = process_image_for_embedding(img_path)  # Implement or use pre-trained model
        text_emb = model.encode(text)
        # Here you would combine or concatenate embeddings
        combined_emb = np.concatenate([img_emb, text_emb])
        return combined_emb
    else:
        return model.encode(text)

# Function to handle chat interactions
def moderated_chat(message, history):
    global query_engine, rails
    if query_engine is None:
        return history + [("Please upload a file first.", None)]
    try:
        response = query_engine.query(message)
        full_response = ""
        for chunk in response.response_gen:
            full_response += chunk
            moderated_chunk = rails.generate(context=full_response, prompt=message)
            history.append((message, moderated_chunk))
            yield history
    except Exception as e:
        yield history + [(message, f"An error occurred: {str(e)}")]

#def chat(message, history):
#    global query_engine
#    if query_engine is None:
#        return history + [("Please upload a file first.", None)]
#    try:
        # get the user's message
#        user_message = message
        
        # Assuming query_engine is set up to retrieve relevant documents
#        response = query_engine.query(user_message)
        
        # validate and potentially modify the response using guardrail
#       validated_response = rails.generate(
#            context=response.response,
#            prompt=user_message,
#        )

        #update the chat history with the validated response
 #       history.append((user_message, validated_response))
 #       return history

 #   except Except as e:
 #       return history + [(message, f"Error processing query: {str(e)}")]

# Function to stream responses
def stream_response(message,history):
    global query_engine
    if query_engine is None:
        yield history + [("Please upload a file first.",None)]
        return

    try:
        response = query_engine.query(message)
        partial_response = ""
        for text in response.response_gen:
            partial_response += text
            yield history + [(message,partial_response)]
    except Exception as e:
        yield history + [(message, f"Error processing query: {str(e)}")]

# Create the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Multimodal RAG Chatbot with NeMo Guardrails 🦀 ")
    gr.Markdown("Upload **at least** one document, optional to input HTML URL before Q&A")  
    with gr.Row():
        with gr.Column():
            file_uploader = gr.Files(label="Select files to upload",file_count="multiple")
            url_input = gr.Textbox(label="Enter URL", placeholder="Optional: Enter HTML URL here")
        with gr.Column():
            load_btn = gr.Button("Load Documents & URL")
    load_output = gr.Textbox(label="Processing Status")
    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Enter your question",interactive=True)
    clear = gr.ClearButton(components=[chatbot, msg])

    load_btn.click(load_documents, inputs=[file_uploader, url_input], outputs=load_output)
    msg.submit(moderated_chat, inputs=[msg, chatbot], outputs=chatbot, queue=False)
    clear.click(fn=lambda: ([], ""), outputs=[chatbot, msg])
    

# Launch the Gradio interface
if __name__ == "__main__":
    demo.queue().launch(share=True,debug=True)
