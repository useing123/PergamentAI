from fastapi import FastAPI, WebSocket, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import requests
import json
import openai
from langchain.document_loaders import PDFMinerLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this to specific domains
    allow_credentials=True,
    allow_methods=["*"],  # You can restrict this to specific methods
    allow_headers=["*"],  # You can restrict this to specific headers
)

# Get environment variables
DATABASE_ID = os.getenv('DATABASE_ID')
NOTION_API_KEY = os.getenv('NOTION_API_KEY')
PARENT_PAGE_ID = os.getenv('PARENT_PAGE_ID')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

openai.api_key = OPENAI_API_KEY

headers = {
    "Authorization": f"Bearer {NOTION_API_KEY}",
    "Content-Type": "application/json",
    "Notion-Version": "2022-06-28"
}

chat_history = []
pdf_documents = []

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        chat_history.append(data)

        if not pdf_documents:
            await websocket.send_text("Please upload a PDF file first.")
            continue

        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        vector_store = FAISS.from_documents(pdf_documents, embeddings)
        retriever = vector_store.as_retriever()
        llm = OpenAI(model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)
        qa_chain = load_qa_chain(llm, chain_type="stuff")

        response = qa_chain.run(input_documents=pdf_documents, question=data)
        chat_history.append(response)
        await websocket.send_text(response)

def create_notion_page(content):
    blocks = []
    for line in content.split('\n'):
        if line.startswith("# "):
            blocks.append({
                "object": "block",
                "type": "heading_1",
                "heading_1": {"rich_text": [{"type": "text", "text": {"content": line[2:]}}]}
            })
        elif line.startswith("## "):
            blocks.append({
                "object": "block",
                "type": "heading_2",
                "heading_2": {"rich_text": [{"type": "text", "text": {"content": line[3:]}}]}
            })
        elif line.startswith("### "):
            blocks.append({
                "object": "block",
                "type": "heading_3",
                "heading_3": {"rich_text": [{"type": "text", "text": {"content": line[4:]}}]}
            })
        elif line.startswith("- "):
            blocks.append({
                "object": "block",
                "type": "bulleted_list_item",
                "bulleted_list_item": {"rich_text": [{"type": "text", "text": {"content": line[2:]}}]}
            })
        elif line.startswith("1. "):
            blocks.append({
                "object": "block",
                "type": "numbered_list_item",
                "numbered_list_item": {"rich_text": [{"type": "text", "text": {"content": line[3:]}}]}
            })
        else:
            rich_text = []
            parts = line.split('**')
            for i, part in enumerate(parts):
                if i % 2 == 0:
                    subparts = part.split('*')
                    for j, subpart in enumerate(subparts):
                        if j % 2 == 0:
                            rich_text.append({"type": "text", "text": {"content": subpart}})
                        else:
                            rich_text.append({"type": "text", "text": {"content": subpart, "annotations": {"italic": True}}})
                else:
                    rich_text.append({"type": "text", "text": {"content": part, "annotations": {"bold": True}}})

            blocks.append({"object": "block", "type": "paragraph", "paragraph": {"rich_text": rich_text}})
    
    data = {
        "parent": {"type": "page_id", "page_id": PARENT_PAGE_ID},
        "properties": {
            "title": {
                "title": [
                    {
                        "text": {
                            "content": "Generated Insights"
                        }
                    }
                ]
            }
        },
        "children": blocks
    }

    response = requests.post(
        'https://api.notion.com/v1/pages',
        headers=headers,
        data=json.dumps(data)
    )

    if response.status_code == 200:
        return "Page created successfully!"
    else:
        return f"Failed to create page. Status code: {response.status_code}. Response: {response.text}"

@app.post("/send-to-notion")
async def send_to_notion():
    content = "\n".join(chat_history)
    result = create_notion_page(content)
    if "successfully" in result:
        return JSONResponse(content={"message": result})
    else:
        raise HTTPException(status_code=500, detail=result)

def extract_text_from_pdf(pdf_path: str):
    loader = PDFMinerLoader(pdf_path)
    documents = loader.load()
    return documents

def generate_embeddings_and_retrieve_info(documents):
    try:
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        vector_store = FAISS.from_documents(documents, embeddings)
        retriever = vector_store.as_retriever()
        llm = OpenAI(model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)
        qa_chain = load_qa_chain(llm, chain_type="stuff")

        query = "Please provide a summary of the document in markdown format for notion document with full notes"
        response = qa_chain.run(input_documents=documents, question=query)
        return response
    except Exception as e:
        print(f"Error generating embeddings or retrieving information: {e}")
        return None

# Function to split text into blocks with headings support
def split_text_to_blocks(text, max_length=2000):
    blocks = []
    while len(text) > max_length:
        split_index = text.rfind('\n', 0, max_length)
        if split_index == -1:
            split_index = text.rfind(' ', 0, max_length)
        if split_index == -1:
            split_index = max_length
        
        block = text[:split_index]
        text = text[split_index:].strip()

        # Ensure block ends with a complete sentence
        if not block.endswith('.') and len(text) > 0:
            sentence_end_index = block.rfind('.')
            if sentence_end_index != -1:
                split_index = sentence_end_index + 1
                block = block[:split_index]
                text = text[split_index:].strip()

        blocks.append(block)
    blocks.append(text)
    return blocks

@app.post("/process-pdf/")
async def process_pdf(file: UploadFile = File(...)):
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    file_location = os.path.join(upload_dir, file.filename)
    
    with open(file_location, "wb") as f:
        f.write(await file.read())
    
    documents = extract_text_from_pdf(file_location)
    if not documents:
        raise HTTPException(status_code=400, detail="Failed to extract text from PDF")
    
    global pdf_documents
    pdf_documents = documents

    response = generate_embeddings_and_retrieve_info(documents)
    if not response:
        raise HTTPException(status_code=500, detail="Failed to generate response from the document")
    
    result = create_notion_page(response)
    if "successfully" in result:
        return JSONResponse(content={"message": result})
    else:
        raise HTTPException(status_code=500, detail=result)
