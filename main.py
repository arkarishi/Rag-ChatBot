from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import Union, List
import numpy as np
import cohere

class Agent:
    def __init__(self, api_key):
        self.client = cohere.ClientV2(api_key)
        self.embedding_model = "embed-english-v3.0"
        self.summarisation_model = "command-r-plus-08-2024"
        self.rerank_model = "rerank-english-v2.0"
        self.generating_model = "command-r"
        self.input_document = "search_document"
        self.input_query = "search_query"
        self.splits = None
        self.vectordb = None

    def load_paper(self, file_path, embed=False) -> Union[List[str], str]:
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

    def document_embeddings(self, file_path) -> list:
        self.doc_embeds = self.client.embed(
            texts = self.load_paper(file_path, embed=True),
            model = self.embedding_model,
            input_type= self.input_document,
            embedding_types=["float"]
        ).embeddings.float
        self.vectordb = {
            i:np.array(embedding) for i, embedding in enumerate(self.doc_embeds)
        }
        return self.doc_embeds
    
    def query_embeddings(self, query) -> list:
        self.query_embeds = self.client.embed(
            texts=[query],
            model=self.embedding_model, 
            input_type=self.input_query, 
            embedding_types=["float"]
        ).embeddings.float[0]
        return self.query_embeds
    
    def summarise(self, file_path) -> str:
        """
        implement logic for summarisation 
        """
        message = f"""
        Generate a concise summary for the text: {self.load_paper(file_path)}
        """
        response = self.client.chat(
            model = self.summarisation_model,
            messages=[{"role": "user", "content": message}]
        )
        return response.message.content[0].text

    def rag(self, file_path, query) -> str:
        """
        implement logic for question-answering
        """
        def cosine_similarity(a, b):
            return np.dot(a, b) / np.linalg.norm(a) * np.linalg.norm(b)
        

        document_embeds = self.document_embeddings(file_path)
        query_embeds = self.query_embeddings(query)

        similarities = [cosine_similarity(query_embeds, embedding) for embedding in document_embeds]
        sorted_indices = np.argsort(similarities)[::-1]
        top_indices = sorted_indices[:10]
        top_chunks_after_retrieval = [self.splits[i].page_content for i in top_indices]


        # TODO : Implement Reranking
        # rerank_response = self.client.rerank(
        #     query = query, 
        #     documents=top_chunks_after_retrieval,
        #     top_n=3,
        #     model = self.rerank_model
        # )
        

        preamble = """
        ## Task &amp; Context
        You help people answer their questions and other requests interactively. 
        You will be asked questions related to a particular modern research topic. 
        You will be equipped with a wide range of search engines or similar tools to help you, which you use to research your answer. 
        You should focus on serving the user's needs as best you can, which will be wide-ranging.

        ## Style Guide
        Unless the user asks for a different style of answer, you should answer in full sentences, using proper grammar and spelling.
        """

        documents = [
            {"title": i, 
             "snippet": top_chunks_after_retrieval[i], 
             "data": {"text": top_chunks_after_retrieval[i]}} for i in range(min(len(top_chunks_after_retrieval), 5))
        ]

        response = self.client.chat(
            messages = [{"role": "user", "content": f"{preamble} {query}"}],
            documents = documents,
            # preamble = preamble,
            model = self.summarisation_model,
            temperature=0.3
        )

        return response.message.content[0].text

    
    

    


