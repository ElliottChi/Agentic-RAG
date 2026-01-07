import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import tool, AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# 1. 設定 API Key
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

# 2. 定義工具 (Tools) - 這是 Agent 的手腳
# 使用 @tool 裝飾器，LangChain 會自動把函數說明(Docstring)轉給 LLM 看

@tool
def multiply(a: int, b: int) -> int:
    """將兩個數字相乘。當你需要計算數學時使用這個工具。"""
    print(f"\n[Tool Log] 正在執行乘法: {a} * {b} ...")
    return a * b

@tool
def get_weather(city: str) -> str:
    """查詢某個城市的當前天氣。"""
    print(f"\n[Tool Log] 正在查詢 {city} 的天氣 ...")
    # 這裡我們假裝去查了氣象局 API
    if "台北" in city or "Taipei" in city:
        return "晴天, 25度C"
    elif "新竹" in city:
        return "多雲, 22度C"
    else:
        return "未知天氣"
    
@tool
def get_stock_price(symbol: str):
    """查詢股價。"""
    print(f"\n[Tool log] 正在查詢 {symbol} 的股價")
    if "Apple" in symbol or "AAPL" in symbol:
        return "260 USD"
    elif "Nvidia" in symbol or "NVDA" in symbol:
        return "190 USD"
    else:
        return "未知股價"

tools = [multiply, get_weather, get_stock_price]

# 3. 初始化大腦 (LLM)
# 使用 model="gpt-3.5-turbo" 或 "gpt-4o"
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 4. 建立 Agent
# 這是標準的 Prompt，告訴 LLM：「你是一個助教，你有這些工具可以使用...」
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一個有用的助手。如果遇到你無法回答的問題，請使用工具。"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"), # 這是 LLM 思考過程的筆記本
])

# 建立 Agent 實體 (Brain + Tools + Prompt)
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 5. 實際執行任務
print("--- 開始執行任務 ---")
query = "Apple現在股價多少？"
response = agent_executor.invoke({"input": query})

print(f"\n--- 最終回答 ---\n{response['output']}")