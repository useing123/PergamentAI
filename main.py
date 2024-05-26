from fastapi import FastAPI, WebSocket, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse, FileResponse
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
from langchain.schema import Document
import os
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
import uuid

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
markdown_content = ""


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        chat_history.append(data)

        if "youtube.com/watch" in data:
            video_id = data.split("v=")[1].split("&")[0]
            try:
                transcript = YouTubeTranscriptApi.get_transcript(video_id)
                transcript_text = "\n".join([entry['text'] for entry in transcript])
                data = transcript_text
            except Exception as e:
                await websocket.send_text(f"Failed to fetch transcript: {str(e)}")
                continue

        if not pdf_documents and "youtube.com/watch" not in data:
            await websocket.send_text("Please upload a PDF file first or provide a YouTube video link.")
            continue

        if "youtube.com/watch" in data:
            documents = [Document(page_content=data)]
        else:
            embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
            vector_store = FAISS.from_documents(pdf_documents, embeddings)
            retriever = vector_store.as_retriever()
            llm = OpenAI(model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)
            qa_chain = load_qa_chain(llm, chain_type="stuff")

            response = qa_chain.run(input_documents=pdf_documents, question=data)
            chat_history.append(response)
            await websocket.send_text(response)

        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        vector_store = FAISS.from_documents([Document(page_content=data)], embeddings)
        retriever = vector_store.as_retriever()
        llm = OpenAI(model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)
        qa_chain = load_qa_chain(llm, chain_type="stuff")

        response = qa_chain.run(input_documents=[Document(page_content=data)], question="Please summarize the transcript.")
        chat_history.append(response)
        await websocket.send_text(response)


@app.post("/process-youtube/")
async def process_youtube(youtube_url: str = Form(...)):
    if "youtube.com/watch" not in youtube_url:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")

    video_id = youtube_url.split("v=")[1].split("&")[0]
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = "\n".join([entry['text'] for entry in transcript])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch transcript: {str(e)}")

    chat_history.append(transcript_text)

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vector_store = FAISS.from_documents([Document(page_content=transcript_text)], embeddings)
    retriever = vector_store.as_retriever()
    llm = OpenAI(model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)
    qa_chain = load_qa_chain(llm, chain_type="stuff")

    response = qa_chain.run(input_documents=[Document(page_content=transcript_text)], question="Please summarize the transcript in markdown format.")
    chat_history.append(response)

    global markdown_content
    markdown_content = response

    result = create_notion_page(response)
    if "successfully" in result:
        return JSONResponse(content={"message": result})
    else:
        raise HTTPException(status_code=500, detail=result)


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
    global markdown_content
    markdown_content = content
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

        query = """You're an AI assistant that specializes in creating summaries and document formatting. You excel at condensing information into clear, concise markdown format for easy integration into Notion documents.
Your task is to generate a markdown-formatted summary of a document, including full notes. Ensure the summary is well-organized with proper formatting for easy readability.
Keep in mind to capture the key points of the document while maintaining coherence and clarity in the summary.

Please provide a summary of the document in markdown format for a Notion document with full notes."""
        response = qa_chain.run(input_documents=documents, question=query)
        return response
    except Exception as e:
        print(f"Error generating embeddings or retrieving information: {e}")
        return None


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
    
    global markdown_content
    markdown_content = response
    result = create_notion_page(response)
    if "successfully" in result:
        return JSONResponse(content={"message": result})
    else:
        raise HTTPException(status_code=500, detail=result)


@app.get("/download-markdown/")
async def download_markdown():
    global markdown_content
    if not markdown_content:
        raise HTTPException(status_code=404, detail="No content available to download")
    
    file_id = str(uuid.uuid4())
    file_path = f"downloads/{file_id}.md"
    os.makedirs("downloads", exist_ok=True)
    
    with open(file_path, "w") as f:
        f.write(markdown_content)
    
    return FileResponse(file_path, filename="generated_notes.md")
