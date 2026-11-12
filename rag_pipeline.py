import os
import shutil
import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Configuration paths
DB_FAISS_PATH = "vectorstore/db_faiss"
DATA_PATH = "data/"

st.set_page_config(page_title="PDF RAG Assistant", page_icon="📄", layout="wide")
st.title("📄 Chat with Your PDFs (RAG Pipeline)")

# --- Sidebar: API Key & Document Management ---
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=os.environ.get("OPENAI_API_KEY", ""),
        help="Enter your OpenAI API key (starts with sk-...)"
    )
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    st.divider()
    st.header("📂 Document Upload")
    uploaded_files = st.file_uploader(
        "Upload 2–3 PDF documents",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button("Process & Index PDFs", type="primary"):
        if not api_key:
            st.error("Please enter your OpenAI API key first.")
        elif not uploaded_files:
            st.warning("Please upload at least one PDF file.")
        else:
            with st.spinner("Saving PDFs, chunking text, and building FAISS index..."):
                # Clean old files in data directory so only newly uploaded PDFs are indexed
                if os.path.exists(DATA_PATH):
                    shutil.rmtree(DATA_PATH)
                os.makedirs(DATA_PATH, exist_ok=True)

                for file in uploaded_files:
                    file_path = os.path.join(DATA_PATH, file.name)
                    with open(file_path, "wb") as f:
                        f.write(file.getbuffer())

                # Remove old FAISS vector store
                if os.path.exists(DB_FAISS_PATH):
                    shutil.rmtree(DB_FAISS_PATH)

                # Load, split, embed, and save FAISS index
                loader = PyPDFDirectoryLoader(DATA_PATH)
                docs = loader.load()

                if not docs:
                    st.error("No text could be extracted. Ensure your PDFs contain selectable text.")
                    st.stop()

                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                chunks = text_splitter.split_documents(docs)

                embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
                vectorstore = FAISS.from_documents(chunks, embeddings)

                os.makedirs(os.path.dirname(DB_FAISS_PATH), exist_ok=True)
                vectorstore.save_local(DB_FAISS_PATH)

                # Clear cached resource so the new index loads immediately
                st.cache_resource.clear()
                st.success(f"Indexed {len(chunks)} chunks from {len(uploaded_files)} PDF(s)!")

    if st.button("Clear Index & Chat History"):
        if os.path.exists(DB_FAISS_PATH):
            shutil.rmtree(DB_FAISS_PATH)
        if os.path.exists(DATA_PATH):
            shutil.rmtree(DATA_PATH)
        st.cache_resource.clear()
        st.session_state.messages = []
        st.rerun()

# --- Helpers ---
@st.cache_resource
def load_vectorstore(api_key_check: str):
    if os.path.exists(DB_FAISS_PATH) and api_key_check:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        return FAISS.load_local(
            DB_FAISS_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
    return None

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# --- Main Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("📚 Sources Consulted"):
                for src in message["sources"]:
                    st.write(f"- **{src['file']}** (Page {src['page']})")

# Handle user input
if prompt_input := st.chat_input("Ask a question about your uploaded PDFs..."):
    if not api_key:
        st.error("Please enter your OpenAI API key in the sidebar.")
        st.stop()

    vectorstore = load_vectorstore(api_key)
    if vectorstore is None:
        st.warning("No FAISS index found. Please upload PDFs and click 'Process & Index PDFs' in the sidebar.")
        st.stop()

    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt_input})
    with st.chat_message("user"):
        st.markdown(prompt_input)

    # Generate response using LCEL (no langchain.chains needed)
    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            retrieved_docs = retriever.invoke(prompt_input)

            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
            system_prompt = (
                "You are an intelligent assistant trained to answer questions based strictly on the provided context.\n"
                "If you do not know the answer or if it's not present in the context, say that you cannot find the answer "
                "in the documents. Do not make up information.\n\n"
                "Context:\n{context}"
            )
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}"),
            ])

            rag_chain = (
                {"context": lambda _: format_docs(retrieved_docs), "input": RunnablePassthrough()}
                | prompt_template
                | llm
                | StrOutputParser()
            )
            answer = rag_chain.invoke(prompt_input)

            sources = []
            for doc in retrieved_docs:
                sources.append({
                    "file": os.path.basename(doc.metadata.get("source", "Unknown")),
                    "page": doc.metadata.get("page", 0) + 1
                })

            st.markdown(answer)
            if sources:
                with st.expander("📚 Sources Consulted"):
                    for src in sources:
                        st.write(f"- **{src['file']}** (Page {src['page']})")

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources
            })
