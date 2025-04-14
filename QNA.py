import sys,traceback
import tornado.web
import tornado.ioloop
import os
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
# from langchain.document_loaders import UnstructuredFileLoader
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA



##http://localhost:8885/chatbot?query=what is QBOT?
##http://localhost:8888/startchatbot?

qa_chain = None
def configure_chatbot():
    global qa_chain
    print("welcome")
    GOOGLE_API_KEY = ""
    genai.configure(api_key=GOOGLE_API_KEY)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, google_api_key=GOOGLE_API_KEY, )
    # load the document
    loader = UnstructuredFileLoader(" QUESTION AND ANSWERS.pdf")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(separator='/n',
                                          chunk_size=1000,
                                          chunk_overlap=200)

    text_chunks = text_splitter.split_documents(documents)
    # loading the vector embedding model
    embeddings = HuggingFaceEmbeddings()
    # vector embedding for text chunks
    knowledge_base = FAISS.from_documents(text_chunks, embeddings)
    # chain for aq retrieval
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=knowledge_base.as_retriever())

class QueryParameterChatbotRequestHandler(tornado.web.RequestHandler):
    def get(self):
        try:
            question = str(self.get_argument("query"))
            response = qa_chain.invoke({"query": question})
            answer = response["result"]
            print(answer)
            # answer = str(answer).replace(".",".\n")
            # print(answer)
            self.write(str(answer))
        except Exception as e:
            ex_type, ex, tb = sys.exc_info()
            traceback.print_exception(ex_type, ex, tb)
            print(e)

if __name__ == "__main__":
    configure_chatbot()
    app = tornado.web.Application([
        (r"/chatbot", QueryParameterChatbotRequestHandler)

    ])

    port = 8885
    app.listen(port)
    print(f"Application is ready and listening on port {port}")
    tornado.ioloop.IOLoop.current().start()

