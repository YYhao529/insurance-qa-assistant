import os, qa_config
import shutil

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader, CSVLoader


class VectorStoreBuilder:

    # 构造方法
    def __init__(self):
        # 创建文档分割器
        self.splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", "。", "，", " ", ""],  # 分隔符
            chunk_size=qa_config.CHUNK_SIZE,  # 每块最大字符上限
            chunk_overlap=qa_config.CHUNK_OVERLAP  # 相邻块重叠字符
        )

        # 创建向量模型
        self.embeddings_model = OllamaEmbeddings(model="bge-m3")

    # 构建向量数据库
    def build_store(self):
        # 每次构建前清空旧库
        if os.path.exists(qa_config.VECTORSTORE_PATH):
            shutil.rmtree(qa_config.VECTORSTORE_PATH)

        # 1.文档加载
        # 1-1 读取data目录下的文件
        for file_name in os.listdir(qa_config.DATA_PATH):
            # 1-2 判断文件的类型是否匹配
            file_type = "." + file_name.split(".")[-1].lower()
            if file_type not in qa_config.SUPPORTED_EXTENSIONS:
                continue  # 如果文件类型不匹配，跳过

            # 1-3 拼接文件的完整路径     ./data/insurance_compare.csv
            file_path = qa_config.DATA_PATH + "/" + file_name

            # 1-4 根据文档的类型进行判断
            loader = None
            if file_type == ".txt":
                loader = TextLoader(file_path=file_path, encoding="utf-8")
            elif file_type == ".pdf":
                loader = PyPDFLoader(file_path=file_path)
            elif file_type == ".csv":
                loader = CSVLoader(file_path=file_path, encoding="utf-8")

            # 1-5 文档加载处理
            docs = loader.load()
            # 2.文档拆分
            documents = self.splitter.split_documents(docs)
            # 3.文档向量化存储
            Chroma.from_documents(documents, self.embeddings_model, persist_directory=qa_config.VECTORSTORE_PATH)


    # 检索向量数据库
    def query_store(self,query):
        # 1.连接向量数据库
        db=Chroma(persist_directory=qa_config.VECTORSTORE_PATH,embedding_function=self.embeddings_model)
        # 2.执行查询
        docs = db.similarity_search_with_relevance_scores(query=query, k=qa_config.TOP_K)
        # 3.打印结果
        for doc in docs:
            print(doc)

if __name__ == "__main__":
    vector_store_builder = VectorStoreBuilder()
    # vector_store_builder.build_store()
    vector_store_builder.query_store("如何购买保险")
