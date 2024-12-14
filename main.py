from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
import cohere

class Agent:
    def __init__(self, api_key):
        self.client = cohere.Client(api_key)
        self.model = "embed-english-v3.0"
        self.input_document = "search_document"
        self.input_query = "search_query"
        # self.splits = None
        self.vectordb = None

    def load_paper(self, file_path):
        document =  PyPDFLoader(
            file_path = file_path
        ).load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 200
        )
        self.splits = text_splitter.split_documents(document)
        return [split.page_content for split in self.splits]

    def document_embeddings(self, file_path):
        self.embeddings = self.client.embed(
            texts = self.load_paper(file_path),
            model = self.model,
            input_type= self.input_document,
            embedding_types=["float"]
        ).embeddings.float
        return self.embeddings
    
    def summarise(self):
        """
        implement logic for summarisation 
        """
        pass

    def query(self):
        """
        implement logic for question-answering
        """
        pass

    
    

    


