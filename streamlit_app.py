import os
import re
import hashlib
import streamlit as st
from openai import OpenAI
import chromadb
from pypdf import PdfReader

# ========== 配置 ==========
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

# ========== 升级版 ask 函数：返回答案 + 来源 ==========
def ask_with_source(collection, query, history=None, top_k=2):
    """
    检索相关片段，结合历史对话生成答案，并返回答案和来源
    """
    # 1. 将问题向量化
    q_emb = get_embedding(query)
    # 2. 检索最相关的 top_k 个片段
    results = collection.query(query_embeddings=[q_emb], n_results=top_k)
    chunks = results['documents'][0]
    
    # 3. 构建上下文
    context = "\n\n---\n\n".join(chunks)
    
    # 4. 构建提示词（如果有历史对话，也加进去）
    prompt = f"""你是一个专业的文档问答助手。请根据以下参考资料回答用户的问题。
如果参考资料中没有相关信息，请明确回答“资料中没有提到”。

参考资料：
{context}

"""
    if history:
        prompt += "\n对话历史：\n"
        for item in history:
            prompt += f"用户：{item['user']}\n助手：{item['assistant']}\n"
    
    prompt += f"""
用户当前问题：{query}
答案："""
    
    # 5. 调用大模型
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    answer = resp.choices[0].message.content
    
    # 6. 处理来源（用于显示）
    sources = []
    for idx, chunk in enumerate(chunks):
        preview = chunk[:200] + "..." if len(chunk) > 200 else chunk
        sources.append({
            "index": idx + 1,
            "preview": preview,
            "full": chunk
        })
    
    return answer, sources

# ========== Streamlit UI ==========
st.set_page_config(page_title="📄 PDF智能问答助手", layout="wide")
st.title("📚 PDF 智能问答系统")
st.markdown("上传你的PDF文档，然后像聊天一样提问，AI会从文档中寻找答案，并支持追问。")

# 初始化会话状态
if "messages" not in st.session_state:
    st.session_state.messages = []
if "history" not in st.session_state:
    st.session_state.history = []  # 存储对话历史（用于上下文）
if "sources" not in st.session_state:
    st.session_state.sources = {}  # 存储每条消息对应的来源

# 侧边栏
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

# 显示聊天记录（带来源折叠）
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # 如果是助手消息且有来源，显示来源（可折叠）
        if msg["role"] == "assistant" and idx in st.session_state.sources:
            sources = st.session_state.sources[idx]
            with st.expander("📖 查看答案来源"):
                for src in sources:
                    st.markdown(f"**片段 {src['index']}**：")
                    st.text(src['preview'])

# 底部输入框
if prompt := st.chat_input("在这里输入你的问题"):
    # 检查是否已上传PDF
    if "collection" not in st.session_state or st.session_state["collection"] is None:
        st.warning("请先在左侧上传一个PDF文件。")
        st.stop()
    
    # 显示用户问题
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # 生成回答
    with st.chat_message("assistant"):
        with st.spinner("正在从文档中查找答案..."):
            try:
                # 调用升级版函数，传入历史对话
                answer, sources = ask_with_source(
                    st.session_state["collection"], 
                    prompt, 
                    history=st.session_state.history
                )
                st.markdown(answer)
                
                # 保存消息和来源
                msg_idx = len(st.session_state.messages)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                st.session_state.sources[msg_idx] = sources
                
                # 更新对话历史（用于下次提问的上下文）
                st.session_state.history.append({
                    "user": prompt,
                    "assistant": answer
                })
                # 只保留最近 5 轮对话，避免超出上下文
                if len(st.session_state.history) > 5:
                    st.session_state.history.pop(0)
                
                # 显示来源（可折叠）
                if sources:
                    with st.expander("📖 查看答案来源"):
                        for src in sources:
                            st.markdown(f"**片段 {src['index']}**：")
                            st.text(src['preview'])
            except Exception as e:
                err_msg = f"生成答案时出错：{e}"
                st.error(err_msg)
                st.session_state.messages.append({"role": "assistant", "content": err_msg})
