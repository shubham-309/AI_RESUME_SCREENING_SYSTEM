import pinecone
from langchain.schema import Document
from langchain.embeddings import SentenceTransformerEmbeddings
from pypdf import PdfReader
#from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.vectorstores import Pinecone
from dotenv import load_dotenv

load_dotenv()

def load(pdf):
    text = ""
    pdf_reader = PdfReader(pdf)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def create_doc(user_pdf_list, unique_id):
    docs = []
    for filename in user_pdf_list:
        chunk = load(filename)

        docs.append(Document(
            page_content=chunk,
            metadata={"name": filename.name, "type": filename.type, "size": filename.size,
                      "unique_id": unique_id},
        ))
    return docs

def create_embedding_load_data():
    embedding = OpenAIEmbeddings()
    return embedding

def create_pinecone_index():
    pinecone.create_index("resumeanalyser", dimension=1536, metric="cosine")

def push_to_pinecone(pinecone_apikey, pinecone_environment, pinecone_index_name, embedding, docs):
    pinecone.init(api_key=pinecone_apikey, environment=pinecone_environment)

    Pinecone.from_documents(docs, embedding, index_name=pinecone_index_name)
    print("Done pushing to Pinecone.")

def delete_from_pinecone(pinecone_apikey, pinecone_environment, pinecone_index_name):
    pinecone.init(api_key=pinecone_apikey, environment=pinecone_environment)
    pinecone.delete_index(pinecone_index_name)
    create_pinecone_index()
    print("Done Deleting from Pinecone.")    



def pull_from_pinecone(pinecone_apikey, pinecone_environment, pinecone_index_name, embedding, docs):
    pinecone.init(api_key=pinecone_apikey, environment=pinecone_environment)
    global index
    index = pinecone.Index(index_name=pinecone_index_name)
    return index 
    


def get_summary(doc):
    llm_g = ChatGoogleGenerativeAI(model="gemini-pro")
    chain = load_summarize_chain(llm_g, chain_type="map_reduce")
    summary = chain.run([doc])
    return summary

def similar_doc(query, k, pinecone_apikey, pinecone_environment, pinecone_index_name, embedding, unique_id, docs):
    pinecone.init(api_key=pinecone_apikey, environment=pinecone_environment)
    docsearch = Pinecone.from_existing_index(pinecone_index_name, embedding)
    similar_docs = docsearch.similarity_search_with_score(query, k=int(k))
    return similar_docs