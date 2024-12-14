from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
import cohere

class Agent:
    def __init__(self, api_key):
        self.client = cohere.ClientV2(api_key)
        self.embedding_model = "embed-english-v3.0"
        self.summarisation_model = "command-r-plus-08-2024"
        self.input_document = "search_document"
        self.input_query = "search_query"
        self.splits = None
        self.vectordb = None

    def load_paper(self, file_path, embed=False):
        document =  PyPDFLoader(
            file_path = file_path
        ).load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 200
        )
        self.splits = text_splitter.split_documents(document)
        if embed == True:
            return [split.page_content for split in self.splits]
        else: 
            return " ".join([split.page_content for split in self.splits])

    def document_embeddings(self, file_path):
        self.embeddings = self.client.embed(
            texts = self.load_paper(file_path, embed=True),
            model = self.embedding_model,
            input_type= self.input_document,
            embedding_types=["float"]
        ).embeddings.float
        return self.embeddings
    
    def summarise(self, file_path):
        """
        implement logic for summarisation 
        """
        message = f"Generate a concise summary for the text: {str(self.load_paper(file_path))}"
        response = self.client.chat(
            model = self.summarisation_model,
            messages=[{"role": "user", "content": message}]
        )
        return response.message.content[0].text

    def query(self):
        """
        implement logic for question-answering
        """
        pass

    
    

    


