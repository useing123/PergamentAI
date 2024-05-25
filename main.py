from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
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

# Function to split text into blocks
def split_text_to_blocks(text, max_length=2000):
    blocks = []
    while len(text) > max_length:
        split_index = text.rfind('\n', 0, max_length)
        if split_index == -1:
            split_index = text.rfind(' ', 0, max_length)
        if split_index == -1:
            split_index = max_length
        blocks.append(text[:split_index])
        text = text[split_index:].strip()
    blocks.append(text)
    return blocks

def create_notion_page(content):
    blocks = split_text_to_blocks(content)
    children = [{"object": "block", "type": "paragraph", "paragraph": {"rich_text": [{"type": "text", "text": {"content": block}}]}} for block in blocks]

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
        "children": children
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
    
    response = generate_embeddings_and_retrieve_info(documents)
    if not response:
        raise HTTPException(status_code=500, detail="Failed to generate response from the document")
    
    result = create_notion_page(response)
    if "successfully" in result:
        return JSONResponse(content={"message": result})
    else:
        raise HTTPException(status_code=500, detail=result)