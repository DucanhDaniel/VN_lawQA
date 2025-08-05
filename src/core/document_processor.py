from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import glob 

class DocumentProcessor:
    def __init__(self, file_path = "data/", chunk_size = 1000, chunk_overlap = 20, length_function = len, 
                 verbose = False):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = chunk_size,
            chunk_overlap = chunk_overlap,
            length_function = length_function
        )

        self.file_path = file_path
        self.documents = []

        for file_path in glob.glob(self.file_path + "*.docx"):
            if verbose:
                print(file_path)
            loader = Docx2txtLoader(file_path)
            self.documents.extend(loader.load())

        for file_path in glob.glob(self.file_path + "*.pdf"):
            if verbose:
                print(file_path)
            loader = PyPDFLoader(file_path)
            self.documents.extend(loader.load())
    
        self.splits = self.text_splitter.split_documents(self.documents)

    def get_splits(self):
        return self.splits

test = DocumentProcessor()
print(test.get_splits())