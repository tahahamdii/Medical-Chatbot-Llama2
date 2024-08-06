from src.helper import load_pfd,text_split,download_hugging_face_embeddings
from langchain.vectorstores import Pinecone
import pinecone
from dotenv import load_dotenv
import os
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer



load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')

print(PINECONE_API_KEY)

extracted_data = load_pfd("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()


#Initializing the Pinecone
pc = Pinecone(api_key='c7eca03a-c885-49a2-bbd9-b4bc52fa3157')
pc.list_indexes()
index_name="medicalchatbot"
index = pc.Index(index_name)


#Creating embeddings for Each text chunks and storing


model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode([t.page_content for t in text_chunks])

# Upload documents to the Pinecone index
for i, (text, embedding) in enumerate(zip([t.page_content for t in text_chunks], embeddings)):
    index.upsert(vectors=[(str(i), embedding.tolist(), {'text': text})])



