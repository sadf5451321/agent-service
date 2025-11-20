import asyncio
import os
import urllib.parse
import uuid
from collections.abc import AsyncGenerator

import streamlit as st
from dotenv import load_dotenv
from pydantic import ValidationError

from client import AgentClient, AgentClientError
from schema import ChatHistory, ChatMessage
from schema.task_data import TaskData, TaskDataStatus

import tempfile
import shutil
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader


APP_TITLE = "Agent Service Toolkit"
APP_ICON = "🧰"
USER_ID_COOKIE = "user_id"


def get_or_create_user_id() -> str:
    """Get the user ID from session state or URL parameters, or create a new one if it doesn't exist."""
    # Check if user_id exists in session state
    if USER_ID_COOKIE in st.session_state:
        return st.session_state[USER_ID_COOKIE]

    # Try to get from URL parameters using the new st.query_params
    if USER_ID_COOKIE in st.query_params:
        user_id = st.query_params[USER_ID_COOKIE]
        st.session_state[USER_ID_COOKIE] = user_id
        return user_id

    # Generate a new user_id if not found
    user_id = str(uuid.uuid4())

    # Store in session state for this session
    st.session_state[USER_ID_COOKIE] = user_id

    # Also add to URL parameters so it can be bookmarked/shared
    st.query_params[USER_ID_COOKIE] = user_id

    return user_id


async def main() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=APP_ICON,
        menu_items={},
    )

    # Hide the streamlit upper-right chrome
    st.html(
        """
        <style>
        [data-testid="stStatusWidget"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
            }
        </style>
        """,
    )
    if st.get_option("client.toolbarMode") != "minimal":
        st.set_option("client.toolbarMode", "minimal")
        await asyncio.sleep(0.1)
        st.rerun()

    # Get or create user ID
    user_id = get_or_create_user_id()

    if "agent_client" not in st.session_state:
        load_dotenv()
        agent_url = os.getenv("AGENT_URL")
        if not agent_url:
            host = os.getenv("HOST", "0.0.0.0")
            port = os.getenv("PORT", 8080)
            agent_url = f"http://{host}:{port}"
        
        # Retry connection with exponential backoff
        max_retries = 5
        retry_delay = 2
        connected = False
        
        with st.spinner("Connecting to agent service..."):
            for attempt in range(max_retries):
                try:
                    st.session_state.agent_client = AgentClient(base_url=agent_url)
                    connected = True
                    break
                except AgentClientError as e:
                    if attempt < max_retries - 1:
                        # Wait before retrying (exponential backoff)
                        import time
                        time.sleep(retry_delay * (attempt + 1))
                        continue
                    else:
                        # Last attempt failed
                        st.error(f"Error connecting to agent service at {agent_url}: {e}")
                        st.markdown(
                            f"""
                            **Connection Failed After {max_retries} Attempts**
                            
                            The agent service might still be starting up. Please:
                            1. Wait a few more seconds
                            2. Check if the service is running: `docker compose ps`
                            3. Check service logs: `docker compose logs agent_service`
                            4. Refresh this page to retry
                            """
                        )
                        st.stop()
        
        if not connected:
            st.error("Failed to connect to agent service")
            st.stop()
    agent_client: AgentClient = st.session_state.agent_client

    if "thread_id" not in st.session_state:
        thread_id = st.query_params.get("thread_id")
        if not thread_id:
            thread_id = str(uuid.uuid4())
            messages_list: list[ChatMessage] = []
        else:
            try:
                history = agent_client.get_history(thread_id=thread_id)
                messages_list = history.messages if history else []
            except AgentClientError:
                st.error("No message history found for this Thread ID.")
                messages_list = []
        st.session_state.messages = messages_list
        st.session_state.thread_id = thread_id

    # Config options
    with st.sidebar:
        st.header(f"{APP_ICON} {APP_TITLE}")

        ""
        "Full toolkit for running an AI agent service built with LangGraph, FastAPI and Streamlit"
        ""

        if st.button(":material/chat: New Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.thread_id = str(uuid.uuid4())
            st.rerun()

        with st.popover(":material/settings: Settings", use_container_width=True):
            if agent_client.info and agent_client.info.models:
                model_idx = agent_client.info.models.index(agent_client.info.default_model) if agent_client.info.default_model in agent_client.info.models else 0
                model = st.selectbox("LLM to use", options=agent_client.info.models, index=model_idx)
            else:
                model = None
                st.warning("无法获取模型列表")
            
            if agent_client.info and agent_client.info.agents:
                agent_list = [a.key for a in agent_client.info.agents]
                agent_idx = agent_list.index(agent_client.info.default_agent) if agent_client.info.default_agent in agent_list else 0
                agent_client.agent = st.selectbox(
                    "Agent to use",
                    options=agent_list,
                    index=agent_idx,
                )
            else:
                st.warning("无法获取 Agent 列表")
            
            use_streaming = st.toggle("Stream results", value=True)

            # Display user ID (for debugging or user information)
            st.text_input("User ID (read-only)", value=user_id, disabled=True)

        @st.dialog("Architecture")
        def architecture_dialog() -> None:
            st.image(
                "https://github.com/JoshuaC215/agent-service-toolkit/blob/main/media/agent_architecture.png?raw=true"
            )
            "[View full size on Github](https://github.com/JoshuaC215/agent-service-toolkit/blob/main/media/agent_architecture.png)"
            st.caption(
                "App hosted on [Streamlit Cloud](https://share.streamlit.io/) with FastAPI service running in [Azure](https://learn.microsoft.com/en-us/azure/app-service/)"
            )

        if st.button(":material/schema: Architecture", use_container_width=True):
            architecture_dialog()

        with st.popover(":material/policy: Privacy", use_container_width=True):
            st.write(
                "Prompts, responses and feedback in this app are anonymously recorded and saved to LangSmith for product evaluation and improvement purposes only."
            )

        @st.dialog("Share/resume chat")
        def share_chat_dialog() -> None:
            try:
                session = st.runtime.get_instance()._session_mgr.list_active_sessions()[0]  # type: ignore
                st_base_url = urllib.parse.urlunparse(
                    [session.client.request.protocol, session.client.request.host, "", "", "", ""]  # type: ignore
                )
            except Exception:
                st_base_url = "http://localhost:8501"
            # if it's not localhost, switch to https by default
            if not st_base_url.startswith("https") and "localhost" not in st_base_url:
                st_base_url = st_base_url.replace("http", "https")
            # Include both thread_id and user_id in the URL for sharing to maintain user identity
            chat_url = (
                f"{st_base_url}?thread_id={st.session_state.thread_id}&{USER_ID_COOKIE}={user_id}"
            )
            st.markdown(f"**Chat URL:**\n```text\n{chat_url}\n```")
            st.info("Copy the above URL to share or revisit this chat")

        if st.button(":material/upload: Share/resume chat", use_container_width=True):
            share_chat_dialog()

        "[View the source code](https://github.com/JoshuaC215/agent-service-toolkit)"
        st.caption(
            "Made with :material/favorite: by [Joshua](https://www.linkedin.com/in/joshua-k-carroll/) in Oakland"
        )

        # ========== 文件上传和向量数据库管理 ==========
        with st.expander(":material/upload_file: 上传文件并创建向量数据库", expanded=False):
            st.markdown("### 📁 上传文档")
            st.markdown("支持格式: PDF, DOCX, TXT")
            
            uploaded_files = st.file_uploader(
                "选择文件",
                type=["pdf", "docx", "txt"],
                accept_multiple_files=True,
                help="可以一次上传多个文件"
            )
            
            if uploaded_files:
                st.info(f"已选择 {len(uploaded_files)} 个文件")
                for file in uploaded_files:
                    st.text(f"  • {file.name} ({file.size / 1024:.1f} KB)")
            
            # 数据库配置选项
            st.markdown("### ⚙️ 数据库配置")
            db_name = st.text_input(
                "数据库名称",
                value="chroma_db_uploaded",
                help="向量数据库的存储路径"
            )
            
            chunk_size = st.slider(
                "文本块大小",
                min_value=500,
                max_value=5000,
                value=2000,
                step=500,
                help="每个文本块的最大字符数"
            )
            
            overlap = st.slider(
                "文本块重叠",
                min_value=0,
                max_value=1000,
                value=500,
                step=100,
                help="相邻文本块之间的重叠字符数"
            )
            
            use_local_embeddings = st.toggle(
                "使用本地 Embedding 模型",
                value=False,
                help="如果启用，使用本地模型（需要设置 USE_LOCAL_MODEL=true）"
            )
            
            # 创建数据库按钮
            if st.button("🚀 创建向量数据库", use_container_width=True, type="primary"):
                if not uploaded_files:
                    st.error("请先上传文件！")
                else:
                    await create_vector_db_from_files(
                        uploaded_files=uploaded_files,
                        db_name=db_name,
                        chunk_size=chunk_size,
                        overlap=overlap,
                        use_local_embeddings=use_local_embeddings
                    )
        
        # 数据库选择器
        st.markdown("---")
        with st.popover(":material/storage: 向量数据库管理", use_container_width=True):
            available_dbs = get_available_databases()
            if available_dbs:
                selected_db = st.selectbox(
                    "选择向量数据库",
                    options=available_dbs,
                    index=0,
                    help="选择要使用的向量数据库"
                )
                if st.button("✅ 使用此数据库", use_container_width=True):
                    # 设置环境变量或更新配置
                    os.environ["CHROMA_DB_PATH"] = selected_db
                    st.success(f"已切换到数据库: {selected_db}")
                    st.rerun()
            else:
                st.info("暂无可用的向量数据库")

    # Draw existing messages
    # Draw existing messages
    messages: list[ChatMessage] = st.session_state.messages

    if len(messages) == 0:
        match agent_client.agent:
            case "chatbot":
                WELCOME = "Hello! I'm a simple chatbot. Ask me anything!"
            case "interrupt-agent":
                WELCOME = "Hello! I'm an interrupt agent. Tell me your birthday and I will predict your personality!"
            case "research-assistant":
                WELCOME = "Hello! I'm an AI-powered research assistant with web search and a calculator. Ask me anything!"
            case "rag-assistant":
                WELCOME = """Hello! I'm an AI-powered Company Policy & HR assistant with access to AcmeTech's Employee Handbook.
                I can help you find information about benefits, remote work, time-off policies, company values, and more. Ask me anything!"""
            case _:
                WELCOME = "Hello! I'm an AI agent. Ask me anything!"

        with st.chat_message("ai"):
            st.write(WELCOME)

    # draw_messages() expects an async iterator over messages
    async def amessage_iter() -> AsyncGenerator[ChatMessage, None]:
        for m in messages:
            yield m

    await draw_messages(amessage_iter())

    # Generate new message if the user provided new input
    if user_input := st.chat_input():
        messages.append(ChatMessage(type="human", content=user_input))
        st.chat_message("human").write(user_input)
        try:
            if use_streaming:
                stream = agent_client.astream(
                    message=user_input,
                    model=model,
                    thread_id=st.session_state.thread_id,
                    user_id=user_id,
                )
                await draw_messages(stream, is_new=True)
            else:
                response = await agent_client.ainvoke(
                    message=user_input,
                    model=model,
                    thread_id=st.session_state.thread_id,
                    user_id=user_id,
                )
                messages.append(response)
                st.chat_message("ai").write(response.content)
            st.rerun()  # Clear stale containers
        except AgentClientError as e:
            st.error(f"Error generating response: {e}")
            st.stop()

    # If messages have been generated, show feedback widget
    if len(messages) > 0 and st.session_state.last_message:
        with st.session_state.last_message:
            await handle_feedback()


async def draw_messages(
    messages_agen: AsyncGenerator[ChatMessage | str, None],
    is_new: bool = False,
) -> None:
    """
    Draws a set of chat messages - either replaying existing messages
    or streaming new ones.

    This function has additional logic to handle streaming tokens and tool calls.
    - Use a placeholder container to render streaming tokens as they arrive.
    - Use a status container to render tool calls. Track the tool inputs and outputs
      and update the status container accordingly.

    The function also needs to track the last message container in session state
    since later messages can draw to the same container. This is also used for
    drawing the feedback widget in the latest chat message.

    Args:
        messages_aiter: An async iterator over messages to draw.
        is_new: Whether the messages are new or not.
    """

    # Keep track of the last message container
    last_message_type = None
    st.session_state.last_message = None

    # Placeholder for intermediate streaming tokens
    streaming_content = ""
    streaming_placeholder = None

    # Iterate over the messages and draw them
    while msg := await anext(messages_agen, None):
        # str message represents an intermediate token being streamed
        if isinstance(msg, str):
            # If placeholder is empty, this is the first token of a new message
            # being streamed. We need to do setup.
            if not streaming_placeholder:
                if last_message_type != "ai":
                    last_message_type = "ai"
                    st.session_state.last_message = st.chat_message("ai")
                if st.session_state.last_message:
                    with st.session_state.last_message:
                        streaming_placeholder = st.empty()
                else:
                    streaming_placeholder = st.empty()

            streaming_content += msg
            streaming_placeholder.write(streaming_content)
            continue
        if not isinstance(msg, ChatMessage):
            st.error(f"Unexpected message type: {type(msg)}")
            st.write(msg)
            st.stop()

        match msg.type:
            # A message from the user, the easiest case
            case "human":
                last_message_type = "human"
                st.chat_message("human").write(msg.content)

            # A message from the agent is the most complex case, since we need to
            # handle streaming tokens and tool calls.
            case "ai":
                # If we're rendering new messages, store the message in session state
                if is_new:
                    st.session_state.messages.append(msg)

                # If the last message type was not AI, create a new chat message
                if last_message_type != "ai":
                    last_message_type = "ai"
                    st.session_state.last_message = st.chat_message("ai")

                if st.session_state.last_message:
                    with st.session_state.last_message:
                        # If the message has content, write it out.
                        # Reset the streaming variables to prepare for the next message.
                        if msg.content:
                            if streaming_placeholder:
                                streaming_placeholder.write(msg.content)
                                streaming_content = ""
                                streaming_placeholder = None
                            else:
                                st.write(msg.content)

                        if msg.tool_calls:
                            # Create a status container for each tool call and store the
                            # status container by ID to ensure results are mapped to the
                            # correct status container.
                            call_results = {}
                            for tool_call in msg.tool_calls:
                                # Use different labels for transfer vs regular tool calls
                                if "transfer_to" in tool_call["name"]:
                                    label = f"""💼 Sub Agent: {tool_call["name"]}"""
                                else:
                                    label = f"""🛠️ Tool Call: {tool_call["name"]}"""

                                status = st.status(
                                    label,
                                    state="running" if is_new else "complete",
                                )
                                call_results[tool_call["id"]] = status

                            # Expect one ToolMessage for each tool call.
                            for tool_call in msg.tool_calls:
                                if "transfer_to" in tool_call["name"]:
                                    status = call_results[tool_call["id"]]
                                    status.update(expanded=True)
                                    await handle_sub_agent_msgs(messages_agen, status, is_new)
                                    break

                                # Only non-transfer tool calls reach this point
                                status = call_results[tool_call["id"]]
                                status.write("Input:")
                                status.write(tool_call["args"])
                                tool_result_raw = await anext(messages_agen)
                                
                                if isinstance(tool_result_raw, str):
                                    st.error(f"Unexpected string message: {tool_result_raw}")
                                    continue
                                
                                tool_result: ChatMessage = tool_result_raw

                                if tool_result.type != "tool":
                                    st.error(f"Unexpected ChatMessage type: {tool_result.type}")
                                    st.write(tool_result)
                                    st.stop()

                                # Record the message if it's new, and update the correct
                                # status container with the result
                                if is_new:
                                    st.session_state.messages.append(tool_result)
                                if tool_result.tool_call_id:
                                    status = call_results[tool_result.tool_call_id]
                                status.write("Output:")
                                status.write(tool_result.content)
                                status.update(state="complete")

            case "custom":
                # CustomData example used by the bg-task-agent
                # See:
                # - src/agents/utils.py CustomData
                # - src/agents/bg_task_agent/task.py
                try:
                    task_data: TaskData = TaskData.model_validate(msg.custom_data)
                except ValidationError:
                    st.error("Unexpected CustomData message received from agent")
                    st.write(msg.custom_data)
                    st.stop()

                if is_new:
                    st.session_state.messages.append(msg)

                if last_message_type != "task":
                    last_message_type = "task"
                    st.session_state.last_message = st.chat_message(
                        name="task", avatar=":material/manufacturing:"
                    )
                    with st.session_state.last_message:
                        status = TaskDataStatus()

                status.add_and_draw_task_data(task_data)

            # In case of an unexpected message type, log an error and stop
            case _:
                st.error(f"Unexpected ChatMessage type: {msg.type}")
                st.write(msg)
                st.stop()


async def handle_feedback() -> None:
    """Draws a feedback widget and records feedback from the user."""

    # Keep track of last feedback sent to avoid sending duplicates
    if "last_feedback" not in st.session_state:
        st.session_state.last_feedback = (None, None)

    # Check if there are messages and if the last message has a run_id
    if not st.session_state.messages:
        return
    
    latest_message = st.session_state.messages[-1]
    latest_run_id = latest_message.run_id if hasattr(latest_message, 'run_id') else None
    
    # Only show feedback widget if run_id is available
    if not latest_run_id:
        return
    
    feedback = st.feedback("stars", key=latest_run_id)

    # If the feedback value or run ID has changed, send a new feedback record
    if feedback is not None and (latest_run_id, feedback) != st.session_state.last_feedback:
        # Normalize the feedback value (an index) to a score between 0 and 1
        normalized_score = (feedback + 1) / 5.0

        agent_client: AgentClient = st.session_state.agent_client
        try:
            await agent_client.acreate_feedback(
                run_id=latest_run_id,
                key="human-feedback-stars",
                score=normalized_score,
                kwargs={"comment": "In-line human feedback"},
            )
        except AgentClientError as e:
            st.error(f"Error recording feedback: {e}")
            st.stop()
        st.session_state.last_feedback = (latest_run_id, feedback)
        st.toast("Feedback recorded", icon=":material/reviews:")


async def handle_sub_agent_msgs(messages_agen, status, is_new):
    """
    This function segregates agent output into a status container.
    It handles all messages after the initial tool call message
    until it reaches the final AI message.

    Enhanced to support nested multi-agent hierarchies with handoff back messages.

    Args:
        messages_agen: Async generator of messages
        status: the status container for the current agent
        is_new: Whether messages are new or replayed
    """
    nested_popovers = {}

    # looking for the transfer Success tool call message
    first_msg = await anext(messages_agen)
    if is_new:
        st.session_state.messages.append(first_msg)

    # Continue reading until we get an explicit handoff back
    while True:
        # Read next message
        sub_msg = await anext(messages_agen)

        # this should only happen is skip_stream flag is removed
        # if isinstance(sub_msg, str):
        #     continue

        if is_new:
            st.session_state.messages.append(sub_msg)

        # Handle tool results with nested popovers
        if sub_msg.type == "tool" and sub_msg.tool_call_id in nested_popovers:
            popover = nested_popovers[sub_msg.tool_call_id]
            popover.write("**Output:**")
            popover.write(sub_msg.content)
            continue

        # Handle transfer_back_to tool calls - these indicate a sub-agent is returning control
        if (
            hasattr(sub_msg, "tool_calls")
            and sub_msg.tool_calls
            and any("transfer_back_to" in tc.get("name", "") for tc in sub_msg.tool_calls)
        ):
            # Process transfer_back_to tool calls
            for tc in sub_msg.tool_calls:
                if "transfer_back_to" in tc.get("name", ""):
                    # Read the corresponding tool result
                    transfer_result = await anext(messages_agen)
                    if is_new:
                        st.session_state.messages.append(transfer_result)

            # After processing transfer back, we're done with this agent
            if status:
                status.update(state="complete")
            break

        # Display content and tool calls in the same nested status
        if status:
            if sub_msg.content:
                status.write(sub_msg.content)

            if hasattr(sub_msg, "tool_calls") and sub_msg.tool_calls:
                for tc in sub_msg.tool_calls:
                    # Check if this is a nested transfer/delegate
                    if "transfer_to" in tc["name"]:
                        # Create a nested status container for the sub-agent
                        nested_status = status.status(
                            f"""💼 Sub Agent: {tc["name"]}""",
                            state="running" if is_new else "complete",
                            expanded=True,
                        )

                        # Recursively handle sub-agents of this sub-agent
                        await handle_sub_agent_msgs(messages_agen, nested_status, is_new)
                    else:
                        # Regular tool call - create popover
                        popover = status.popover(f"{tc['name']}", icon="🛠️")
                        popover.write(f"**Tool:** {tc['name']}")
                        popover.write("**Input:**")
                        popover.write(tc["args"])
                        # Store the popover reference using the tool call ID
                        nested_popovers[tc["id"]] = popover


async def create_vector_db_from_files(
    uploaded_files: list,
    db_name: str = "./chroma_db_uploaded",
    chunk_size: int = 2000,
    overlap: int = 500,
    use_local_embeddings: bool = False,
) -> None:
    """
    从上传的文件创建向量数据库
    
    Args:
        uploaded_files: Streamlit 上传的文件列表
        db_name: 数据库名称/路径
        chunk_size: 文本块大小
        overlap: 文本块重叠
        use_local_embeddings: 是否使用本地 embedding 模型
    """
    from dotenv import load_dotenv
    
    load_dotenv()
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # 创建临时目录存储上传的文件
        with tempfile.TemporaryDirectory() as temp_dir:
            status_text.text("📥 保存上传的文件...")
            progress_bar.progress(10)
            
            # 保存所有上传的文件到临时目录
            saved_files = []
            for i, uploaded_file in enumerate(uploaded_files):
                file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                saved_files.append(file_path)
            
            status_text.text("🔧 初始化 Embedding 模型...")
            progress_bar.progress(20)
            
            # 获取 embeddings
            if use_local_embeddings:
                from langchain_community.embeddings import HuggingFaceEmbeddings
                cache_folder = os.path.join(os.getcwd(), "embedding.model")
                model_name = os.getenv("LOCAL_MODEL_NAME", "BAAI/bge-small-en-v1.5")
                os.environ.setdefault("HF_HUB_OFFLINE", "1")
                embeddings = HuggingFaceEmbeddings(
                    model_name=model_name,
                    cache_folder=cache_folder,
                    model_kwargs={"device": "cpu"},
                    encode_kwargs={"normalize_embeddings": True},
                )
            else:
                from langchain_openai import OpenAIEmbeddings
                embeddings = OpenAIEmbeddings()
            
            status_text.text("🗄️ 创建向量数据库...")
            progress_bar.progress(30)
            
            # 如果数据库已存在，删除它
            if os.path.exists(db_name):
                shutil.rmtree(db_name)
                status_text.text(f"🗑️ 删除现有数据库: {db_name}")
            
            # 创建 Chroma 数据库
            chroma = Chroma(
                embedding_function=embeddings,
                persist_directory=db_name,
            )
            
            # 初始化文本分割器
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=overlap
            )
            
            # 处理每个文件
            total_chunks = 0
            total_files = len(saved_files)
            
            for file_idx, file_path in enumerate(saved_files):
                filename = os.path.basename(file_path)
                status_text.text(f"📄 处理文件 {file_idx + 1}/{total_files}: {filename}")
                progress = 30 + int((file_idx / total_files) * 60)
                progress_bar.progress(progress)
                
                try:
                    # 根据文件类型选择加载器
                    if filename.endswith(".pdf"):
                        loader = PyPDFLoader(file_path)
                    elif filename.endswith(".docx"):
                        loader = Docx2txtLoader(file_path)
                    elif filename.endswith(".txt"):
                        try:
                            loader = TextLoader(file_path, encoding="utf-8")
                        except UnicodeDecodeError:
                            try:
                                loader = TextLoader(file_path, encoding="gbk")
                            except UnicodeDecodeError:
                                loader = TextLoader(file_path, encoding="latin-1")
                    else:
                        st.warning(f"跳过不支持的文件类型: {filename}")
                        continue
                    
                    # 加载并分割文档
                    documents = loader.load()
                    chunks = text_splitter.split_documents(documents)
                    
                    # 添加到向量数据库
                    if chunks:
                        chroma.add_documents(chunks)
                        total_chunks += len(chunks)
                        st.success(f"✅ {filename}: 添加了 {len(chunks)} 个文本块")
                    
                except Exception as e:
                    st.error(f"❌ 处理文件 {filename} 时出错: {str(e)}")
                    continue
            
            progress_bar.progress(100)
            status_text.text("✅ 向量数据库创建完成！")
            
            st.success(f"""
            **向量数据库创建成功！**
            
            - 📁 数据库路径: `{db_name}`
            - 📄 处理文件数: {total_files}
            - 📝 总文本块数: {total_chunks}
            - 🔧 Embedding 模型: {'本地模型' if use_local_embeddings else 'OpenAI'}
            
            现在你可以在"向量数据库管理"中选择使用此数据库。
            """)
            
            # 更新环境变量
            os.environ["CHROMA_DB_PATH"] = db_name
            st.session_state["last_created_db"] = db_name
            
    except Exception as e:
        st.error(f"❌ 创建向量数据库时出错: {str(e)}")
        import traceback
        st.code(traceback.format_exc())


def get_available_databases() -> list[str]:
    """
    获取可用的向量数据库列表
    
    Returns:
        数据库路径列表
    """
    databases = []
    
    # 检查常见的数据库路径
    common_paths = [
        "./chroma_db",
        "./chroma_db_mixed",
        "./chroma_db_uploaded",
        "./qdrant_db",
    ]
    
    for db_path in common_paths:
        if os.path.exists(db_path):
            # 检查是否是有效的数据库目录
            if os.path.isdir(db_path):
                # ChromaDB 通常有这些文件/目录
                if any(
                    os.path.exists(os.path.join(db_path, item))
                    for item in ["chroma.sqlite3", "chroma.sqlite3-wal", "index"]
                ):
                    databases.append(db_path)
                # Qdrant 数据库
                elif os.path.exists(os.path.join(db_path, "config.json")):
                    databases.append(db_path)
    
    # 也检查项目根目录下的其他可能的数据库
    project_root = Path(__file__).parent.parent.parent
    for item in project_root.iterdir():
        if item.is_dir() and ("chroma" in item.name.lower() or "qdrant" in item.name.lower()):
            if item.name not in databases:
                db_path = str(item)
                if os.path.exists(os.path.join(db_path, "chroma.sqlite3")) or os.path.exists(
                    os.path.join(db_path, "config.json")
                ):
                    databases.append(db_path)
    
    return sorted(databases) if databases else []


if __name__ == "__main__":
    asyncio.run(main())
