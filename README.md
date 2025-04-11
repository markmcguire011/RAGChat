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
│   ├── ingestion/ 
│   │   ├── __init__.py
│   │   ├── loader.py 
│   │   ├── processor.py 
│   │   └── indexer.py  
│   │
│   ├── retrieval/    
│   │   ├── __init__.py
│   │   ├── retriever.py  
│   │   └── ranker.py    
│   │
│   ├── llm/     
│   │   ├── __init__.py
│   │   ├── model.py     
│   │   └── prompt.py    
│   │
│   └── chat/         
│       ├── __init__.py
│       ├── history.py 
│       └── bot.py      
│
├── data/             
│   ├── raw/        
│   ├── processed/      
│   └── embeddings/        
│
├── api/                
│   ├── __init__.py
│   ├── routes.py        
│   └── middleware.py    
│
├── ui/                
│   ├── web/           
│   │   ├── static/
│   │   └── templates/
│   └── cli/              
│       └── commands.py
│
├── tests/           
│   ├── __init__.py
│   ├── test_ingestion.py
│   ├── test_retrieval.py
│   └── test_chat.py
│
├── scripts/                
│   ├── ingest_docs.py     
│   └── evaluate.py           
│
├── requirements.txt         
├── .env
├── README.md        
└── .gitignore     
```