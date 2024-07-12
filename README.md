## Local Search with RAG Using Google Drive Integration ##

## Project Description:
This project is designed to create a FastAPI application that interfaces with Google Drive to retrieve files, process them, and utilize Mistral AI to answer user queries based on the content of these files. The application allows setting a specific Google Drive folder to pull files from and enables querying through a web interface.

## Features:
1. Google Drive Integration: Retrieve files from a specified Google Drive folder.
2. PDF Text Extraction: Extract text content from PDF files.
3. Semantic Search: Use FAISS for efficient similarity search.
4. AI-Powered Query Answering: Leverage Mistral AI to answer user queries
based on the extracted content.

## Prerequisites:
1. Python 3.8 or higher
2. Google Cloud account with Drive API enabled
3. Mistral AI account with API key
4. Langchain
5. HuggingFace Token

## Setup Instructions:

1. Install Dependencies:
First, ensure you have Python installed. Then, install the required libraries:
``%pip install --upgrade google-api-python-client google-auth-httplib2 google-auth-oauthlib langchain mistralai fastapi uvicorn chromadb faiss-cpu unstructured langchain-community ipywidgets huggingface_hub nest_asyncio``

## Google Cloud Setup:

## Create a Google Cloud Project:
1. Go to the [Google Cloud Console](https://console.cloud.google.com/).
2. Create a new project.
3. Enable the Google Drive API for your project.
4. Create service account credentials and download the JSON key file.
5. Place the service account JSON file: in your project directory and name it `service_account_file.json`.

## Mistral AI Setup:
1. Register at Mistral AI and obtain an API key.
   
## Running the Application:

Run the FastAPI application using Uvicorn:

``uvicorn main:app --host 0.0.0.0 --port 80``

This command will start the FastAPI server, and you can access the application at
`http://127.0.0.1:8000`. 

## Using the Application:

1. Open your browser and go to `http://127.0.0.1:8000`.
2. Set the Google Drive folder ID using the form.
3. Submit queries through the query form.
   
## Future Scope:
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
