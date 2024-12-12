import langchain 
from langchain.document_loaders import ArxivLoader
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
import cohere

class Agent:
    def __init__(self, api_key):
        self.client = cohere.Client(api_key)
        self.model = "embed-english-v3.0"
        self.input_document = "search_document"
        self.input_query = "search_query"
        self.splits = None
        self.vectordb = None

    def load_paper(self, link):
        document =  ArxivLoader(query=link).load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 200
        )
        self.splits = text_splitter.split_documents(document)

    def document_embeddings(self):
        self.embeddings = self.client.embed(
            texts = self.splits,
            model = self.model
            input_type= self.input_document
        ).embeddings
        self.vector_db = Chroma()
    
    def summarise(self):
        """
        implement logic for summarisation 
        """
        pass

    def query(self):
        """
        implement logic for question-answering
        """

    
    

    


