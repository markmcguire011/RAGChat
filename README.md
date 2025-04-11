# RAG Chatbot with ChromaDB and LangChain

A Retrieval-Augmented Generation (RAG) chatbot that uses ChromaDB for vector storage and LangChain for LLM integration.

## Project Structure
```
rag-chatbot/
│
├── app/                      
│   ├── __init__.py
│   ├── main.py              
│   ├── config.py             
│   │
│   ├── database/            
│   │   ├── __init__.py
│   │   ├── chroma_client.py
│   │   └── vector_store.py 
│   │
│   ├── retrieval/    
│   │   ├── __init__.py
│   │   └── retriever.py   
│   │
│   ├── llm/     
│   │   ├── __init__.py
│   │   ├── model.py     
│   │   └── prompt.py    
│   │
│   └── chat/         
│       ├── __init__.py
│       └── bot.py      
│
├── data/             
│   ├── raw/        
│   ├── processed/      
│   └── embeddings/        
│
│
├── tests/           
│   ├── __init__.py
│   ├── test_retrieval.py
│   └── test_chat.py       
│
├── requirements.txt         
├── .env
├── README.md        
└── .gitignore     
```

## Running the Application

### Vector Store Demo

To test the vector store functionality and see how documents are retrieved:

```bash
python -m app.main --demo
```

This will:
- Load sample documents from the `data/raw/` directory
- Run test queries against the vector store
- Display the retrieved documents and their relevance scores

### Interactive Chat

To start an interactive chat session with the RAG chatbot:

```bash
python -m app.main --chat
```

This will:
- Initialize the vector store with your documents
- Start an interactive command-line chat interface
- Retrieve relevant documents for each query
- Generate responses based on the retrieved context

### Help

To see all available options:

```bash
python -m app.main --help
```