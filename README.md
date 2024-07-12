Local Search with RAG Using Google Drive Integration
Project Description:
This project is designed to create a FastAPI application that interfaces with Google Drive to retrieve files, process them, and utilize Mistral AI to answer user queries based on the content of these files. The application allows setting a specific Google Drive folder to pull files from and enables querying through a web interface.
Features:
● Google Drive Integration: Retrieve files from a specified Google Drive folder.
● PDF Text Extraction: Extract text content from PDF files.
● Semantic Search: Use FAISS for efficient similarity search.
● AI-Powered Query Answering: Leverage Mistral AI to answer user queries
based on the extracted content.
Prerequisites:
● Python 3.8 or higher
● Google Cloud account with Drive API enabled
● Mistral AI account with API key
● Langchain
● HuggingFace Token
Setup Instructions:
1. Install Dependencies:
First, ensure you have Python installed. Then, install the required libraries:
%pip install --upgrade google-api-python-client google-auth-httplib2 google-auth-oauthlib langchain mistralai fastapi uvicorn chromadb faiss-cpu unstructured langchain-community ipywidgets huggingface_hub nest_asyncio
 2. Google Cloud Setup:
1. Create a Google Cloud Project:
● Go to the [Google Cloud Console](https://console.cloud.google.com/).
● Create a new project.
● Enable the Google Drive API for your project.
● Create service account credentials and download the JSON key file.
2. Place the service account JSON file: in your project directory and name it `service_account_file.json`.
3. Mistral AI Setup:
● Register at Mistral AI and obtain an API key. 4. Project Structure:
Organize your project directory as follows:
5. HTML File (`index.html`):
Create a simple HTML file to set the folder ID and submit queries:
  project/
│
├── app.py
├── index.html
└── service_account_file.json
<!DOCTYPE html> <html lang="en"> <head>
   <meta charset="UTF-8">
   <meta name="viewport" content="width=device-width, initial-scale=1.0">
   <title>Local Search with RAG</title>
</head>
<body>
   <div class="container">
       <h1>Local Search with RAG</h1>
 
         <form id="folderForm">
           <label for="folderId">Enter Folder ID:</label>
           <input type="text" id="folderId" name="folderId" required>
           <button type="submit">Set Folder ID</button>
       </form>
       <form id="queryForm">
           <label for="query">Enter Query:</label>
           <input type="text" id="query" name="query" required>
           <button type="submit">Submit Query</button>
</form>
       <div id="response"></div>
   </div>
   <script>
       document.getElementById('folderForm').addEventListener('submit', async
function(e) {
           e.preventDefault();
           const folderId = document.getElementById('folderId').value;
           const response = await fetch('/set_folder/', {
               method: 'POST',
               headers: {
                   'Content-Type': 'application/x-www-form-urlencoded',
               },
               body: new URLSearchParams({
                   'folder_id_input': folderId,
}) });
           const result = await response.json();
           alert(result.message);
       });
       document.getElementById('queryForm').addEventListener('submit', async
function(e) {
           e.preventDefault();
           const query = document.getElementById('query').value;
           const response = await fetch('/query', {
               method: 'POST',
               headers: {
                   'Content-Type': 'application/json',
               },
               body: JSON.stringify({ query: query })
           });
           const result = await response.json();

             document.getElementById('response').innerText = result.answer ||
result.detail;
});
   </script>
</body>
</html>
6. FastAPI Application (`app.py`):
Here's the complete FastAPI application code:
 from fastapi import FastAPI, HTTPException, Form
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List
import os
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import numpy as np
import faiss
import fitz  # PyMuPDF
app = FastAPI()
class QueryRequest(BaseModel):
   query: str
folder_id = None  # Initialize folder_id as None
client = MistralClient(api_key="")  # Update with your Mistral API key
@app.post("/set_folder/")
async def set_folder(folder_id_input: str = Form(...)):
   global folder_id
   folder_id = folder_id_input
   return {"message": "Folder ID set successfully"}
@app.post("/query")
async def query_drive(request: QueryRequest):

     if not folder_id:
       raise HTTPException(status_code=400, detail="Folder ID not set")
   SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
   SERVICE_ACCOUNT_FILE = '/path/to/json/file.json'  # Update this path
   credentials = service_account.Credentials.from_service_account_file(
       SERVICE_ACCOUNT_FILE, scopes=SCOPES)
   service = build('drive', 'v3', credentials=credentials)
   # Function to list files in Google Drive folder
   def list_files_in_folder(service, folder_id):
       query = f"'{folder_id}' in parents and trashed=false"
       try:
           results = service.files().list(q=query, pageSize=10, fields="files(id,
name)").execute()
           items = results.get('files', [])
           if not items:
               print(f"No files found in the folder with ID: {folder_id}")
           return items
       except Exception as e:
           print(f"An error occurred: {e}")
           return []
   # Function to extract text from a PDF file
   def extract_text_from_pdf(file_path):
       doc = fitz.open(file_path)
       text = ""
       for page in doc:
           text += page.get_text()
       return text
   # Retrieve documents from Google Drive
   files = list_files_in_folder(service, folder_id)
   if not files:
       raise ValueError(f"No files found or unable to access the folder with ID:
{folder_id}")
   documents = []
   for file in files:
       file_id = file['id']
       file_name = file['name']

         print(f"Downloading file: {file_name} (ID: {file_id})")
       request = service.files().get_media(fileId=file_id)
       with open(file_name, 'wb') as fh:
           downloader = MediaIoBaseDownload(fh, request)
           done = False
           while done is False:
               status, done = downloader.next_chunk()
       # Extract the content of the file based on its type
       if file_name.endswith('.pdf'):
           text = extract_text_from_pdf(file_name)
       elif file_name.endswith('.txt'):
           with open(file_name, 'r', encoding='utf-8') as f:
               text = f.read()
       else:
           continue  # Skip unsupported file types
       documents.append(text)
   # Split text into chunks
   chunk_size = 2048
   chunks = [doc[i:i + chunk_size] for doc in documents for i in range(0, len(doc),
chunk_size)]
   # Define a function to get text embedding using Mistral AI
   def get_text_embedding(input):
       embeddings_batch_response = client.embeddings(
           model="mistral-embed",
           input=input
       )
       return embeddings_batch_response.data[0].embedding
   # Create embeddings for each text chunk
   text_embeddings = np.array([get_text_embedding(chunk) for chunk in chunks])
   # Load into a vector database (FAISS)
   d = text_embeddings.shape[1]
   index = faiss.IndexFlatL2(d)
   index.add(text_embeddings)
   # Create embeddings for the question
   question = request.query

     question_embeddings = np.array([get_text_embedding(question)])
   # Retrieve similar chunks from the vector database
   D, I = index.search(question_embeddings, k=2)  # Retrieve top 2 similar chunks
   retrieved_chunks = [chunks[i] for i in I.tolist()[0]]
   # Combine context and question in a prompt and generate response
   prompt = f"""
   Context information is below.
   ---------------------
   {' '.join(retrieved_chunks)}
   ---------------------
   Given the context information and not prior knowledge, answer the query.
   Query: {question}
   Answer:
   """
   def run_mistral(user_message, model="mistral-medium-latest"):
       messages = [
           ChatMessage(role="user", content=user_message)
       ]
       chat_response = client.chat(
           model=model,
           messages=messages
       )
       return chat_response.choices[0].message.content
   answer = run_mistral(prompt)
   return {"answer": answer}
@app.get("/", response_class=HTMLResponse)
async def read_root():
   with open("index.html", "r") as file:
       return HTMLResponse(content=file.read(), status_code=200)
if __name__ == "__main__":
   import threading
   import uvicorn
   def run_app():
       uvicorn.run(app, host="0.0.0.0", port=8000)

  thread = threading.Thread(target=run_app, args=(), daemon=True)
thread.start()
7. Running the Application:
Run the FastAPI application using Uvicorn:
uvicorn main:app --host 0.0.0.0 --port 80
This command will start the FastAPI server, and you can access the application at
`http://127.0.0.1:8000`. 8. Using the Application:
1. Open your browser and go to `http://127.0.0.1:8000`. 2. Set the Google Drive folder ID using the form.
3. Submit queries through the query form.
Future Scope:
1. Support for Multiple File Formats:
Currently, the application only supports .txt files. Future enhancements could include support for other file types such as Word documents, Excel spreadsheets, and pdf files. This would require implementing additional file parsing logic.
2. Enhanced Document Processing:
Improving the text extraction capabilities, especially for complex PDF documents with images, tables, and multi-column layouts, would make the application more robust.
3. Incremental Indexing:
Instead of re-indexing all documents every time, implement incremental indexing to add new documents or update existing ones without rebuilding the entire index.
4. Advanced Query Capabilities:
Incorporate natural language processing (NLP) techniques to improve query understanding. This can include handling synonyms, stemming, and more complex query structures.
   
5. User Authentication and Permissions:
Integrate user authentication to restrict access to the application. Implement permissions to control who can set the folder ID, upload documents, and query the system.
6. Scalable Architecture:
For large-scale deployments, consider implementing a microservices architecture and using cloud services to handle increased load. Tools like Kubernetes for orchestration and Docker for containerization can help achieve this.
7. Enhanced Frontend:
Develop a more user-friendly and feature-rich frontend. This could include features like document preview, search result highlighting, and better query input interfaces.
8. Logging and Monitoring:
Implement comprehensive logging and monitoring to track the application's performance and usage. Tools like Prometheus and Grafana can be used to visualize metrics and set up alerts.
9. Integrate with Other AI Models:
In addition to Mistral AI, integrate with other AI models and services to improve the accuracy and relevance of the answers. This could involve using different models for different types of queries or data.
10. Continuous Improvement:
Set up a feedback loop where users can rate the answers they receive. Use this feedback to continuously improve the models and the overall system.
This comprehensive documentation should help you set up, understand, and plan for the future of your FastAPI application. If you encounter any issues or need further assistance, feel free to ask!
