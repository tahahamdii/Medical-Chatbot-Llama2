from flask import Flask,render_template,jsonify,request
from src.helper import download_hugging_face_embeddings
from langchain.vectorstores import Pinecone
import pinecone
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from src.prompot import *
from pinecone import Pinecone

import os

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')

embeddings = download_hugging_face_embeddings()

pc = Pinecone(api_key='c7eca03a-c885-49a2-bbd9-b4bc52fa3157')
pc.list_indexes()
index_name="medicalchatbot"
index = pc.Index(index_name)

#Loading the vecotr with the new update langchain pinecone a retour
vector_store = PineconeVectorStore(index=index,embedding=embeddings)

PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs={"prompt": PROMPT}

llm=CTransformers(model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
                  model_type="llama",
                  config={'max_new_tokens':512,
                          'temperature':0.8
                          })

qa= vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 1, "score_threshold": 0.5},
)
chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=qa,
    return_source_documents=True,
    chain_type_kwargs={"prompt":PROMPT})


@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get",methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result=chain({"query": input})
    print("response : ", result["result"])
    return str(result["result"])

if __name__ == '__main__':
    app.run(debug= True)