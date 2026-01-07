# agent_logic.py
from langchain.agents import tool, AgentExecutor, create_tool_calling_agent
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from rag_chain import get_initialized_chain


@tool
def search_traffic_law(query: str) -> str:
    """
    當使用者詢問關於「交通法規」、「罰款金額」、「違規記點」或「道路交通管理處罰條例」的問題時，
    務必使用此工具來查詢正確的法律資訊。
    輸入應該是使用者的完整問題。
    """
    chain = get_initialized_chain()
    result = chain.invoke({"input": query})
    
    # 節省Token
    return result["answer"]

# 2. 定義計算機工具
@tool
def calculate_fine(expression: str) -> str:
    """
    當使用者需要計算多筆罰款的總額時使用。
    輸入這是一個數學表達式，例如 "1800 + 500"。
    """
    try:
        # 使用 eval 簡單實作
        return str(eval(expression))
    except:
        return "計算錯誤"

# 3. 初始化 Agent
def get_traffic_agent():
    # 使用 GPT-4o-mini 作為大腦 (支援 Tool Calling)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # 將工具放入列表
    tools = [search_traffic_law, calculate_fine]
    
    # 設計 Agent 的 Prompt (包含 Router 的邏輯)
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "你是一個聰明的交通法規助理。你的任務是協助使用者解決交通法律問題。\n"
         "【思考策略】\n"
         "1. 如果使用者是在閒聊 (例如打招呼、問天氣)，請直接親切回覆，不要呼叫工具。\n"
         "2. 如果問題涉及法規 (例如闖紅燈罰多少)，請務必呼叫 'search_traffic_law'。\n"
         "3. 如果使用者問總金額 (例如闖紅燈加沒戴安全帽)，請先查出個別罰款，再呼叫 'calculate_fine' 計算總和。\n"
         "不要憑空捏造法律條文。"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"), # Agent思考過程的筆記本
    ])
    
    # 建立 Tool Calling Agent
    agent = create_tool_calling_agent(llm, tools, prompt)
    
    # 建立執行器 (Executor)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    return agent_executor