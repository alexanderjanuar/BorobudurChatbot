import streamlit as st
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import AIMessage, HumanMessage
import time

# Suggestion pills for new users
SUGGESTION_PILLS = [
    "Apa itu Candi Borobudur?",
    "Kapan Candi Borobudur dibangun?",
    "Siapa yang membangun Borobudur?",
    "Bagaimana sejarah penemuan Borobudur?",
    "Apa makna relief di Borobudur?",
    "Berapa tinggi Candi Borobudur?",
    "Apa fungsi Candi Borobudur?",
    "Bagaimana struktur Candi Borobudur?",
    "Apa yang dimaksud dengan stupa?",
    "Mengapa Borobudur menjadi warisan dunia?",
    "Bagaimana cara berkunjung ke Borobudur?",
    "Apa legenda tentang Borobudur?"
]

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Halo nama saya Bori, saya adalah asisten virtual yang siap membantu menjawab pertanyaan seputar Candi Borobudur. Silakan tanyakan apapun yang ingin kamu ketahui!"}]

if "groq_key" not in st.session_state:
    st.session_state["groq_key"] = None

if "selected_pill" not in st.session_state:
    st.session_state["selected_pill"] = None
    
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
    st.session_state["text_data"] = TextLoader("Data/passage.txt",encoding="utf8").load()

def process_data():
    text_split = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap = 200)
    st.session_state["passage_split"] = text_split.split_documents(st.session_state["text_data"])

def instantiate_embedding():
    st.session_state["indosentencebert_embeddings"] = HuggingFaceEmbeddings(model_name="firqaaa/indo-sentence-bert-base")

def create_retriever():
    db_passages = FAISS.from_documents(st.session_state["passage_split"], embedding=st.session_state["indosentencebert_embeddings"])

    retriever_passages = db_passages.as_retriever(
        search_type="similarity", search_kwargs={"k": 5, "include_metadata": True}
    )

    st.session_state["retriever"] = retriever_passages

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
    llm = ChatGroq(temperature=0.4, model_name="llama-3.1-8b-instant",max_tokens=256,api_key=st.session_state["groq_key"])

    # Create a retriever that understands the chat history using the LLM, compression retriever, and the prompt template.
    st.session_state["history_aware_retriver"] = create_history_aware_retriever(llm, st.session_state["retriever"], contextualize_q_prompt)

def create_chain():
    create_history_retriever()
    # Define a system prompt for the assistant, named "Bori", who specializes in answering questions
    system_prompt = (
        "Kamu adalah asisten bernama Bori, singkatan dari *Borobudur Story*, yang bertugas menjawab pertanyaan seputar Candi Borobudur. "
        "Gunakan potongan konteks yang diberikan untuk membantu menjawab pertanyaan "
        "Jika kamu tidak mengetahui jawabannya, katakan bahwa kamu tidak tahu. Jawaban harus singkat, maksimal tiga kalimat, dan jangan menjawab pertanyaan yang tidak berkaitan dengan Candi Borobudur "
        "Jawablah dengan nada yang ramah dan bersahabat, seperti sedang mengobrol santai namun informatif."
        "\n\n"
        "{context}"  
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),              # System instructions (how Bori should behave)
            MessagesPlaceholder("chat_history"),    # Placeholder for the chat history
            ("human", "{input}"),                   # Placeholder for the user's latest question
        ]
    )

    #Instantiate LLM
    llm = ChatGroq(temperature=0.4, model_name="llama-3.1-8b-instant",max_tokens=256,api_key=st.session_state["groq_key"])

    # Create a chain for answering questions based on the context retrieved.
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # Combine the history-aware retriever and the question-answer chain to form a retrieval-augmented generation (RAG) chain.
    st.session_state["rag_chain"] = create_retrieval_chain(st.session_state["history_aware_retriver"], question_answer_chain)

def stream_data(text):
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.02)

def process_user_input(user_input):
    """Process user input and generate response"""
    if not st.session_state["groq_key"]:
        st.info("Please add your API key to continue.")
        st.stop()
    
    start_inference_time = time.time()
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)
    
    msg = st.session_state["rag_chain"].invoke({"input": user_input, "chat_history": st.session_state["chat_history"]})
    st.session_state.messages.append({"role": "assistant", "content": msg['answer']})
    st.session_state["chat_history"].extend(
        [
            HumanMessage(content=user_input),
            AIMessage(content=msg["answer"]),
        ]
    )
    
    end_inference_time = time.time()
    inference_time = end_inference_time - start_inference_time
    
    with st.chat_message("assistant"):
        response = st.write_stream(stream_data(msg['answer']))
        st.info(f"Inference time: {inference_time:.2f} seconds.")

st.set_page_config(
    page_title="Borobudur Chatbot",
    page_icon="🗿",
    layout="wide",
    initial_sidebar_state="expanded",
)

with st.sidebar:
    st.session_state["groq_key"] = st.text_input("Groq API Key", type="password")
    st.divider()
    if st.session_state["groq_key"]:
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
            st.success('Retriever has been created', icon="✅")

            with st.spinner("Creating Conversatation Chain"):
                create_chain()
            st.success('Conversation Chain has been created, The bot is good to go', icon="✅")
    "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://github.com/alexanderjanuar/BorobudurChatbot)"

st.header("🗿 BoroChat : Your Borobudur Companion",divider="rainbow")

# Display chat messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

# Show suggestion pills only if there are no user messages yet (first time user)
show_suggestions = len([msg for msg in st.session_state.messages if msg["role"] == "user"]) == 0

if show_suggestions and st.session_state.get("rag_chain"):
    st.markdown("### 💡 Pertanyaan yang sering ditanyakan:")
    st.markdown("*Klik salah satu pertanyaan di bawah untuk memulai percakapan*")
    
    # Create columns for better layout
    cols = st.columns(3)
    
    for i, suggestion in enumerate(SUGGESTION_PILLS):
        col_idx = i % 3
        with cols[col_idx]:
            if st.button(suggestion, key=f"pill_{i}", use_container_width=True):
                st.session_state["selected_pill"] = suggestion
                st.rerun()

# Handle chat input
if prompt := st.chat_input("Tanyakan sesuatu tentang Candi Borobudur..."):
    process_user_input(prompt)

# Handle pill selection
if st.session_state.get("selected_pill"):
    process_user_input(st.session_state["selected_pill"])
    st.session_state["selected_pill"] = None  # Clear the selection