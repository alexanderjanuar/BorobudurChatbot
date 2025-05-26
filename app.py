import streamlit as st
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.retrievers import MergerRetriever
from langchain_community.document_transformers import EmbeddingsClusteringFilter,EmbeddingsRedundantFilter
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_groq import ChatGroq
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import AIMessage, HumanMessage
import time

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

if "groq_key" not in st.session_state:
    st.session_state["groq_key"] = None
    st.session_state["gemini_key"] = None

    
if "wiki_data" not in st.session_state:
    st.session_state["wiki_data"] = None
    st.session_state["text_data"] = None

if "wiki_split" not in st.session_state:
    st.session_state["wiki_split"] = None
    st.session_state["passage_split"] = None

if "indosentencebert_embeddings" not in st.session_state:
    st.session_state["indosentencebert_embeddings"] = None
    st.session_state["gte_embeddings"] = None
    st.session_state["filter_embeddings"] = None

if "retriever" not in st.session_state:
    st.session_state["retriever"] = None

if "history_aware_retriver" not in st.session_state:
    st.session_state["history_aware_retriver"] = None

if "rag_chain" not in st.session_state:
    st.session_state["rag_chain"] = None

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

def load_data():
    st.session_state["wiki_data"] = WikipediaLoader(query="Borobudur", load_max_docs=1,doc_content_chars_max = 5000,lang='id').load()
    st.session_state["text_data"] = TextLoader("passages.txt",encoding="utf8").load()

def process_data():
    text_split = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap = 200)
    st.session_state["wiki_split"] = text_split.split_documents(st.session_state["wiki_data"])
    st.session_state["passage_split"] = text_split.split_documents(st.session_state["text_data"])

def instantiate_embedding():
    st.session_state["indosentencebert_embeddings"] = HuggingFaceEmbeddings(model_name="firqaaa/indo-sentence-bert-base")
    st.session_state["gte_embeddings"] = HuggingFaceEmbeddings(model_name="Alibaba-NLP/gte-multilingual-base", model_kwargs={'trust_remote_code': True})
    st.session_state["filter_embeddings"] = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key = st.session_state["gemini_key"])

def create_retriever():
    db_wiki = FAISS.from_documents(st.session_state["wiki_split"], embedding=st.session_state["indosentencebert_embeddings"])
    db_passages = FAISS.from_documents(st.session_state["passage_split"], embedding=st.session_state["gte_embeddings"])

    # Define 2 diff retrievers with 2 diff embeddings and diff search type.
    retriever_wiki = db_wiki.as_retriever(
        search_type="similarity", search_kwargs={"k": 5, "include_metadata": True}
    )
    retriever_passages = db_passages.as_retriever(
        search_type="mmr", search_kwargs={"k": 5, "include_metadata": True}
    )

    # The Lord of the Retrievers will hold the output of both retrievers and can be used as any other
    lotr = MergerRetriever(retrievers=[retriever_wiki, retriever_passages])

    # Remove redundant results from both retrievers using yet another embedding.
    # Using multiples embeddings in diff steps could help reduce biases.
    filter = EmbeddingsRedundantFilter(embeddings=st.session_state["filter_embeddings"])
    pipeline = DocumentCompressorPipeline(transformers=[filter])
    compression_retriever = ContextualCompressionRetriever(base_compressor=pipeline, base_retriever=lotr)

    st.session_state["retriever"] = ContextualCompressionRetriever(base_compressor=compression_retriever, base_retriever=compression_retriever)

def create_history_retriever():
    # Define a prompt to turn a question with chat history context
    # into a clear, standalone question without using the history.
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    # Create a prompt template that combines system instructions, chat history, and user input.
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),  # Placeholder for chat history
            ("human", "{input}"),  # Placeholder for the user's latest question
        ]
    )

    #Instantiate LLM
    llm = ChatGroq(temperature=0.5, model_name="llama-3.1-8b-instant",max_tokens=256,api_key=st.session_state["groq_key"])

    # Create a retriever that understands the chat history using the LLM, compression retriever, and the prompt template.
    st.session_state["history_aware_retriver"] = create_history_aware_retriever(llm, st.session_state["retriever"], contextualize_q_prompt)

def create_chain():
    create_history_retriever()
    # Define a system prompt for the assistant, named "Bori", who specializes in answering questions
    system_prompt = (
        "You are an assistant named Bori that stands for Borobudur Story for question-answering tasks about Candi Borobudur."
        "Use the following pieces of retrieved context to help your answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise and don't answer questions not related to Candi Borobudur."
        "Dont Answer Question that does not related to candi borobudur"
        "\n\n"
        "{context}"  # Placeholder for the retrieved context relevant to the question.
    )

    # Create a chat prompt template that combines the system instructions, chat history, and user input.
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),              # System instructions (how Bori should behave)
            MessagesPlaceholder("chat_history"),    # Placeholder for the chat history
            ("human", "{input}"),                   # Placeholder for the user's latest question
        ]
    )

    #Instantiate LLM
    llm = ChatGroq(temperature=0.5, model_name="llama-3.1-8b-instant",max_tokens=256,api_key=st.session_state["groq_key"])

    # Create a chain for answering questions based on the context retrieved.
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # Combine the history-aware retriever and the question-answer chain to form a retrieval-augmented generation (RAG) chain.
    st.session_state["rag_chain"] = create_retrieval_chain(st.session_state["history_aware_retriver"], question_answer_chain)

def stream_data(text):
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.02)

st.set_page_config(
    page_title="Borobudur Chatbot",
    page_icon="🗿",
    layout="wide",
    initial_sidebar_state="expanded",
)

with st.sidebar:
    st.session_state["groq_key"] = st.text_input("Groq API Key", type="password")
    st.session_state["gemini_key"] = st.text_input("Gemini API Key", type="password")
    st.divider()
    if st.session_state["groq_key"] and st.session_state["gemini_key"]:
        if not st.session_state["rag_chain"]:
            with st.spinner("Loading Data"):
                load_data()
            st.success('Data has been loaded', icon="✅")

            with st.spinner("Splitting Data"):
                process_data()
            st.success('Data has been Splited', icon="✅")

            with st.spinner("Instantiate Embedding"):
                instantiate_embedding()
            st.success('Embedding has been instantiate', icon="✅")

            with st.spinner("Creating Retriever"):
                create_retriever()
            st.success('MergerRetriever & Reranking retriever has been created', icon="✅")

            with st.spinner("Creating Conversatation Chain"):
                create_chain()
            st.success('Conversation Chain has been created, The bot is good to go', icon="✅")
    "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://github.com/alexanderjanuar/BorobudurChatbot)"

st.header("🗿 BoroChat : Your Borobudur Companion",divider="rainbow")

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

if prompt := st.chat_input():
    if not st.session_state["groq_key"]:
        st.info("Please add your API key to continue.")
        st.stop()
    
    start_inference_time = time.time()  # Catat waktu awal inferensi
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    msg = st.session_state["rag_chain"].invoke({"input": prompt, "chat_history": st.session_state["chat_history"]})
    st.session_state.messages.append({"role": "assistant", "content": msg['answer']})
    st.session_state["chat_history"].extend(
        [
            HumanMessage(content=prompt),
            AIMessage(content=msg["answer"]),
        ]
    )
    end_inference_time = time.time()
    inference_time = end_inference_time - start_inference_time
    with st.chat_message("assistant"):
        response = st.write_stream(stream_data(msg['answer']))
        st.info(f"Inference time: {inference_time:.2f} seconds.")
