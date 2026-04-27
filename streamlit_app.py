import os
import re
import hashlib
import streamlit as st
from openai import OpenAI
import chromadb
from pypdf import PdfReader

# ========== 配置 ==========
# 临时方案：直接写死你的 API Key（部署后可以改成 st.secrets）
API_KEY = st.secrets["API_KEY"]
BASE_URL = st.secrets["BASE_URL"]
EMBEDDING_MODEL = "BAAI/bge-large-zh-v1.5"
CHAT_MODEL = "Qwen/Qwen2.5-7B-Instruct"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ========== 辅助函数 ==========
def get_embedding(text):
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return response.data[0].embedding

def split_text_into_chunks(text, max_chars=500):
    paragraphs = re.split(r'\n\s*\n', text)
    chunks = []
    current_chunk = ""
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(para) > max_chars:
            for i in range(0, len(para), max_chars):
                chunks.append(para[i:i+max_chars])
            continue
        if len(current_chunk) + len(para) + 2 <= max_chars:
            if current_chunk:
                current_chunk += "\n\n" + para
            else:
                current_chunk = para
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = para
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n\n"
    return full_text

def get_file_md5(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def get_or_create_collection(pdf_path, persist_dir="./chroma_db"):
    chroma_client = chromadb.PersistentClient(path=persist_dir)
    file_md5 = get_file_md5(pdf_path)
    collection_name = f"pdf_{file_md5}"
    try:
        collection = chroma_client.get_collection(name=collection_name)
        return collection, False
    except Exception:
        text = extract_text_from_pdf(pdf_path)
        if not text.strip():
            raise ValueError("PDF中没有提取到任何文字，可能是扫描件。")
        chunks = split_text_into_chunks(text)
        collection = chroma_client.create_collection(name=collection_name)
        for i, chunk in enumerate(chunks):
            emb = get_embedding(chunk)
            collection.add(ids=[str(i)], embeddings=[emb], documents=[chunk])
        return collection, True

def ask(collection, query, top_k=2):
    q_emb = get_embedding(query)
    results = collection.query(query_embeddings=[q_emb], n_results=top_k)
    chunks = results['documents'][0]
    context = "\n\n".join(chunks)
    prompt = f"""根据下列资料回答问题。如果资料中没有相关信息，请明确回答“资料中没有提到”。

参考资料：
{context}

问题：{query}
答案："""
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return resp.choices[0].message.content

# ========== Streamlit UI ==========
st.set_page_config(page_title="📄 PDF智能问答助手", layout="wide")
st.title("📚 PDF 智能问答系统")
st.markdown("上传你的PDF文档，然后像聊天一样提问，AI会从文档中寻找答案。")

with st.sidebar:
    st.header("⚙️ 控制台")
    uploaded_file = st.file_uploader("上传 PDF 文件", type=["pdf"])
    if uploaded_file is not None:
        temp_pdf_path = f"./temp_{uploaded_file.name}"
        with open(temp_pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.session_state["pdf_path"] = temp_pdf_path
        st.session_state["pdf_name"] = uploaded_file.name
        with st.spinner("正在处理文档，请稍候..."):
            try:
                collection, is_new = get_or_create_collection(temp_pdf_path)
                st.session_state["collection"] = collection
                if is_new:
                    st.success(f"✅ 已成功处理新文档：{uploaded_file.name}")
                else:
                    st.success(f"✅ 已从缓存加载文档：{uploaded_file.name}")
            except Exception as e:
                st.error(f"文档处理失败：{e}")
                st.session_state["collection"] = None
    else:
        st.info("👈 请从左侧上传一个PDF文件开始。")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("在这里输入你的问题"):
    if "collection" not in st.session_state or st.session_state["collection"] is None:
        st.warning("请先在左侧上传一个PDF文件。")
        st.stop()
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("assistant"):
        with st.spinner("正在思考答案..."):
            try:
                answer = ask(st.session_state["collection"], prompt)
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                err_msg = f"生成答案时出错：{e}"
                st.error(err_msg)
                st.session_state.messages.append({"role": "assistant", "content": err_msg})
