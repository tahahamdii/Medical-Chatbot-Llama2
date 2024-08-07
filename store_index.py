from langchain_pinecone import PineconeVectorStore

from src.helper import load_pfd,text_split,download_hugging_face_embeddings
from langchain.vectorstores import Pinecone as LangchainPinecone
from pinecone import Pinecone

import pinecone
from dotenv import load_dotenv
import os
from sentence_transformers import SentenceTransformer



load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')

print(PINECONE_API_KEY)

extracted_data = load_pfd("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

#Initializing the Pinecone
pc = Pinecone(api_key='c7eca03a-c885-49a2-bbd9-b4bc52fa3157')

index_name="medicalchatbot"
index = pc.Index(index_name)
print(index.describe_index_stats())
#Creating embeddings for Each text chunks and storing

#docsearch = LangchainPinecone.from_texts([t.page_content for t in text_chunks], embeddings, index_name=index_name)
#Creating Embeddings for Each of The Text Chunks & storing
texts = ["Tonight, I call on the Senate to: Pass the Freedom to Vote Act.", "ne of the most serious constitutional responsibilities a President has is nominating someone to serve on the United States Supreme Court.", "One of our nation’s top legal minds, who will continue Justice Breyer’s legacy of excellence."]
docsearch = PineconeVectorStore.from_texts(texts, embedding = embeddings, index_name=index_name)
# docsearch = Pinecone.from_texts(texts, embedding = embeddings, index_name=index_name)
docsearch.add_texts([t.page_content for t in text_chunks])
embedded_texts = embeddings.embed_documents([t.page_content for t in text_chunks])