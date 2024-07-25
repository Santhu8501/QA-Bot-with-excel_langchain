from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import streamlit as st
import pandas as pd


# preprocessing the excel data using pandas like dropping irrelavent rows and columns and treating null values
def data_preprocessing(file_path):
    df = pd.read_csv(file_path)
    df_new = df.drop(df.iloc[:, 1:6], axis=1)
    df_new = df_new.drop(df_new.index[0:1])
    df_new = df_new.fillna(0)

    # creating a new file to save the cleaned data
    new_file_path = 'D:\\santhosh\\practice models\\langchain_practice_llm\\data_new.csv'
    df_new.to_csv(new_file_path, index=False)
    print('data preprocessing is successful')

# initiating the function
data_preprocessing('D:\santhosh\practice models\langchain_practice_llm\Capitalaccountreceipts_0.csv')


# declaring the parameters for LLM
n_gpu_layers = 16
n_batch = 16
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
DB_FAISS_PATH = 'vectorstore/db_faiss'


# Loading the LLM model
def load_llm():
    # Load the locally downloaded model here
    llm = LlamaCpp(
        # cache=True,
        model_path="D:\models\Phi-3-mini-4k-instruct-fp16.gguf",
        temperature=0,  # mistral-7b-instruct-v0.2.Q5_K_M.gguf
        top_p=1,
        n_ctx=2000,
        n_gpu_layers=n_gpu_layers,  # Phi-3-mini-4k-instruct-fp16.gguf
        n_batch=n_batch,
        use_mmap=True,
        streaming=False,
        use_mlock=True,  # force to keeo the model in ram
        max_tokens=1000,
        callback_manager=callback_manager,
        verbose=False)
    return llm


# loading the processed CSV file
csv_file_path = 'D:\santhosh\practice models\langchain_practice_llm\data_new.csv'

#loading the csv to console
loader = CSVLoader(file_path=csv_file_path, encoding="utf-8", csv_args={'delimiter': ','})
data = loader.load()
print('cccc',data)

# initiating the embedding model and vector DB
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})
db = FAISS.from_documents(data, embeddings)
# db.save_local(DB_FAISS_PATH)


# initialising a function to make a retrieval chain
def generate(query):

    retriever=db.as_retriever()
    chain = ConversationalRetrievalChain.from_llm(llm=load_llm(), retriever=retriever)
    history = []
    result = chain({"question": query, "chat_history": history})
    answer = result['answer']
    return answer



# using the streamlit library to make an interface to intreact wil application
def main():
    st.title("Chat with your excel Data")
    text_input = st.text_input("Hit with your query")
    if st.button("Ask query"):
        if len(text_input)>0:
            st.info('"Your Query:'+ text_input)
            answer = generate(text_input)
            st.success(answer)
        else:
            answer = 'Type something and hit ask query'
            st.success(answer)

# To run streamlit type command in terminal "streamlit run file_name.py"


# running the mail file
if __name__ == "__main__":
    main()