import uuid
import chromadb
import pandas as pd

class Portfolio:
    def __init__(self, file_path="project/static/test_data.csv"):
        self.file_path = file_path
        self.data = pd.read_csv(file_path)
        self.chromadb_client = chromadb.PersistentClient("vector_store")
        self.collection = self.chromadb_client.get_or_create_collection(name="portfolio")
    
    def load_portfolio(self):
        if not self.collection.count():
            for _, row in self.data.iterrows():
                self.collection.add(
                    documents=row["Techstack"], 
                    metadatas={"links" : row["Links"]},
                    ids=[str(uuid.uuid4())]
                    )  
            
    def query_links(self, skills):
        return self.collection.query(query_texts=skills, n_results=2).get('metadatas', [])


