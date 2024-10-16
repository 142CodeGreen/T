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
def load_documents(file_objs, urls=None):
    global index, query_engine
    try:
        if not file_objs:
            return "Error: No files selected."
            
        all_texts = []  # Accumulate text from all files
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

        images = [f for f in file_objs if f.name.lower().endswith(('.png', '.jpg', '.jpeg'))]
        texts = all_texts  # From your existing extraction
    
        index = create_multimodal_index(texts, images)
        query_engine = index.as_query_engine()
        
      for file_obj in file_objs:  # Iterate through file objects  # to process correct file type
            file_extension = os.path.splitext(file_obj.name)[1].lower()
            if file_extension in file_extractor:
                extractor = file_extractor[file_extension]
                if file_extension in (".jpg", ".png"):  # Process images
                    extracted_text = extract_text_from_image(file_obj)  # Implement extract_text_from_image 
                else:
                    extracted_text = extractor.extract(file_obj)
                all_texts.extend(extracted_text)  # Add extracted text to the list
            else:
                print(f"Warning: Unsupported file type: {file_extension}")
        

        # file_paths = get_files_from_input(file_objs)
        #documents = []
        #for file_path in file_paths:
        #    directory = os.path.dirname(file_path)
        #    documents.extend(SimpleDirectoryReader(input_files=[file_path]).load_data())

        #if not documents:
        #    return f"No documents found in the selected files."


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
        index = VectorStoreIndex.from_documents(multimodal_docs, storage_context=storage_context)
        return index

        # Create the index from the documents
        index = VectorStoreIndex.from_documents(all_texts, embedding=embeddings) # Create index inside the function after documents are loaded

        # Create the query engine after the index is created
        query_engine = index.as_query_engine()
        return f"Successfully loaded {len(documents)} documents from {len(file_paths)} files."
    except Exception as e:
        return f"Error loading documents: {str(e)}"

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
        # Extract text from the webpage
        text = soup.get_text()
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
def chat(message,history):
    global query_engine
    if query_engine is None:
        return history + [("Please upload a file first.",None)]
    try:
        #modification for nemo guardrails ( next three rows)
        user_message = {"role":"user","content":message}
        response = rails.generate(messages=[user_message])
        return history + [(message,response['content'])]
    except Exception as e:
        return history + [(message,f"Error processing query: {str(e)}")]

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
def process_multimodal_uploads(image, text, file):
    # This function would process the inputs as needed. Here's just a placeholder
    # response reflecting what's been uploaded.
    return f"Processed: Image: {'Yes' if image else 'No'}, Text: {text}, File: {file.name if file else 'No file'}"

with gr.Blocks() as demo:
  gr.Markdown("# Multimodal RAG Chatbot 🦀 ")
  gr.Markdown("Upload **at least** one document, optional to input HTML URL before Q&A")  

  with gr.Row():
    with gr.Column():  # Group file_uploader and url_input on the left
      file_uploader = gr.File(label="Select files to upload", file_count="multiple")
      url_input = gr.Textbox(label="🔗 Paste a URL (optional)", placeholder="Enter a HTML URL here")
    load_btn = gr.Button("Upload input (PDF,CSV,Docx,Pptx,JPG,PNG,website)")  # Place load_btn on the right

  load_output = gr.Textbox(label="Load Status")

  chatbot = gr.Chatbot()
  msg = gr.Textbox(label="Enter your question",interactive=True)
  clear = gr.Button("Clear")

# Set up event handler (Event handlers should be defined within the 'with gr.Blocks() as demo:' block)
  load_btn.click(process_multimodal_uploads,inputs=[file_uploader, url_input], outputs=[load_output])
  msg.submit(stream_response, inputs=[msg, chatbot], outputs=[chatbot]) # Use submit button instead of msg
  clear.click(lambda: None, None, chatbot, queue=False)

# Launch the Gradio interface
if __name__ == "__main__":
    demo.queue().launch(share=True,debug=True)
