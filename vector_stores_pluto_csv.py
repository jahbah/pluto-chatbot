from langchain_core.documents import Document
from langchain_chroma.vectorstores import Chroma
import pandas as pd
# from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_openai.embeddings import OpenAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()
# from uuid import uuid4

file_path =os.path.join("data","Pluto_FAQ.csv")
df = pd.read_csv(file_path)

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
db_path = "./chroma_langchain_pluto_csv"
add_docs = not os.path.exists(db_path)

docs =[]
if add_docs:
    for i, row in df.iterrows():
        document = Document(
            page_content = str(row['QUESTION'])+" "+str(row['ANSWER'])
            # metadata = {"Created_by": row["Added_by"] , "Created_at":row["Date_added"]},
            # id= str(uuid4())
        )
        # ids.append(str(id))
        docs.append(document)

vector_store = Chroma(
    collection_name="pluto_faqs",
    persist_directory=db_path,
    embedding_function=embeddings
)
if add_docs:
    print(f"Adding {len(docs)} documents into vector store.")
    vector_store.add_documents(documents=docs)
    print(f"finished injesting file: {file_path}")

retriever = vector_store.as_retriever(search_kwargs={'k': 5})



