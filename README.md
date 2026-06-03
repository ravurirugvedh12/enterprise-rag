# Enterprise RAG Application

A full-stack Retrieval-Augmented Generation (RAG) application using LangChain, FastAPI, ChromaDB, and OpenAI. It allows you to upload internal knowledge documents (PDF, DOCX, TXT, MD) and ask questions against them, providing accurate answers with source citations.

## Features

- **Document Ingestion:** Upload and index PDF, Word (DOCX), text, and Markdown files.
- **Vector Search:** Uses ChromaDB for fast similarity search and OpenAI embeddings (`text-embedding-3-small`).
- **RAG Pipeline:** Powered by LangChain, utilizing `gpt-4o` for generating precise answers based *only* on the provided context.
- **Source Citations:** Every answer includes exact source citations (file name and page number) with excerpts from the original text.
- **Premium Frontend:** A beautiful, responsive, dark-mode glassmorphism UI in a single HTML file.
- **Dynamic Settings:** Hot-swap your OpenAI API key and model (GPT-4o, GPT-3.5 Turbo, etc.) directly from the UI without restarting the server.

## Setup Instructions

### 1. Backend Setup

Open a terminal and navigate to the backend directory:

```bash
cd backend
```

Create a virtual environment (recommended):

```bash
python -m venv venv

# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### 2. Environment Variables

Copy the `.env.example` file to `.env`:

```bash
cp .env.example .env
```

Open `.env` and add your OpenAI API Key. Alternatively, you can start the server without it and configure it via the settings gear icon in the UI later.

```env
OPENAI_API_KEY=sk-...your-key-here...
```

### 3. Run the Backend Server

Start the FastAPI server:

```bash
uvicorn main:app --reload
```

The API will run at `http://localhost:8000`.

### 4. Open the Frontend

You don't need a web server for the frontend! Simply open `frontend/index.html` in your web browser.

Double-click the file, or if you have a local server tool (like VS Code Live Server), you can use that.

## Usage

1. Open the UI in your browser.
2. (Optional) Click the **Settings (⚙️)** icon in the top right to set your API key if you didn't put it in the `.env` file.
3. Drag and drop PDF, DOCX, TXT, or MD files into the **Upload Documents** area on the left sidebar.
4. Wait for them to process. They will appear in the "Indexed Files" list.
5. Type a question in the chat box at the bottom and hit Enter!
