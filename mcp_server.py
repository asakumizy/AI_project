"""
MCP Server - 集成向量数据库查询、天气API和Tavily搜索
"""
import os
import httpx
from typing import Optional

# 获取当前脚本所在目录的 .env 文件
from dotenv import load_dotenv
script_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(script_dir, '.env')
load_dotenv(env_path)  # 加载环境变量

from mcp.server import Server
from mcp.types import Tool, TextContent, Resource
from langchain_community.vectorstores import Milvus
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_openai import ChatOpenAI
from pymilvus import connections, utility

# ==================== 初始化 ====================

# 从环境变量获取 API 密钥（必须从 .env 加载）
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_API_BASE = os.environ.get("OPENAI_API_BASE") or "https://dashscope.aliyuncs.com/compatible-mode/v1"
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
AMAP_API_KEY = os.environ.get("AMAP_API_KEY")

# 验证并设置环境变量（供 LangChain 使用）
if DASHSCOPE_API_KEY:
    os.environ['DASHSCOPE_API_KEY'] = DASHSCOPE_API_KEY
    print(f"✅ DASHSCOPE_API_KEY: {DASHSCOPE_API_KEY[:8]}...")
else:
    print("❌ DASHSCOPE_API_KEY 未设置")

if OPENAI_API_KEY:
    os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
    print(f"✅ OPENAI_API_KEY: {OPENAI_API_KEY[:8]}...")
else:
    print("❌ OPENAI_API_KEY 未设置")

if TAVILY_API_KEY:
    os.environ['TAVILY_API_KEY'] = TAVILY_API_KEY
    print(f"✅ TAVILY_API_KEY: {TAVILY_API_KEY[:8]}...")
else:
    print("❌ TAVILY_API_KEY 未设置")

if AMAP_API_KEY:
    os.environ['AMAP_API_KEY'] = AMAP_API_KEY
    print(f"✅ AMAP_API_KEY: {AMAP_API_KEY[:8]}...")
else:
    print("❌ AMAP_API_KEY 未设置")

if OPENAI_API_BASE:
    os.environ['OPENAI_API_BASE'] = OPENAI_API_BASE
    print(f"✅ OPENAI_API_BASE: {OPENAI_API_BASE}")
# 创建MCP Server实例
app = Server("multi-tool-server")

# ==================== 向量数据库模块 ====================

class VectorDBManager:
    """向量数据库管理器 - 支持通用知识库查询"""
    
    def __init__(self, collection_name: str = "knowledge_base"):
        self.embeddings = None
        self.vectorstore = None
        self.llm = None
        self.retriever = None
        self.collection_name = collection_name
        self._initialized = False
    
    def initialize(self):
        """初始化向量数据库连接"""
        if self._initialized:
            return True
            
        try:
            # 初始化嵌入模型
            self.embeddings = DashScopeEmbeddings(model="text-embedding-v2")
            
            # 初始化大语言模型
            self.llm = ChatOpenAI(
                model="qwen-plus",
                temperature=0.3,
                max_tokens=1000
            )
            
            # 连接Milvus服务
            connections.connect(host="localhost", port="19530")
            
            # 检查集合是否存在
            if utility.has_collection(self.collection_name):
                self.vectorstore = Milvus(
                    embedding_function=self.embeddings,
                    connection_args={"host": "localhost", "port": "19530"},
                    collection_name=self.collection_name,
                    metadata_field="metadata",
                )
                self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
                self._initialized = True
                return True
            else:
                return False
        except Exception as e:
            print(f"向量数据库初始化失败: {e}")
            return False
    
    def search(self, query: str, top_k: int = 3) -> tuple[str, list]:
        """在向量数据库中搜索"""
        if not self._initialized:
            if not self.initialize():
                return "向量数据库未初始化或集合不存在", []
        
        try:
            docs = self.retriever.invoke(query)
            context = "\n\n".join([
                f"【相关文档 {i+1}】\n{doc.page_content}"
                for i, doc in enumerate(docs)
            ])
            
            # 构建提示词
            from langchain_core.prompts import PromptTemplate
            prompt_template = PromptTemplate(
                input_variables=["question", "context"],
                template="""你是一位农业领域的顶尖专家，请基于提供的文档片段回答用户的问题。
                请严格按照以下流程：向量数据库检索—>若信息不够完整—>通过Tavily搜索补充信息—>再整合信息思考一遍合理性和真实性->给出答案。
                
                可用文档片段： { 片段1: [农业技术手册-小麦篇] 小麦赤霉病防治关键期为“齐穗至扬花初期”。一旦错过此窗口期，后期打药效果极差。 
                片段2: [植保站2024年4月预警] 建议选用氰烯菌酯、戊唑醇、咪鲜胺等高效低毒药剂。对于抗性较强地区，推荐使用丙硫菌唑。 
                片段3: [农药安全使用规范] 使用戊唑醇时，每亩用量不应超过30毫升，且每个生长季最多使用2次，以防止产生药害或抗药性。 }

                用户问题：{我家小麦马上要扬花抽穗了，预防赤霉病该什么时候打药？可以用戊唑醇吗？有什么注意事项？}

                可用文档片段：
                {context}
                
                用户问题：{question}
                
                请遵循以下原则：
                1. 仅基于提供的文档片段回答，不要编造信息
                2. 如果文档中没有相关信息，请明确说明
                3. 回答要简洁明了，直接回答用户的问题
                4. 如果文档中有多个相关片段，请综合整理后回答
                
                现在请回答："""
            )
            
            prompt_text = prompt_template.format(context=context, question=query)
            response = self.llm.invoke(prompt_text)
            
            return response.content, docs
        except Exception as e:
            return f"搜索失败: {e}", []

# 全局实例
vector_db = VectorDBManager(collection_name="knowledge_base")

# ==================== 天气API模块 ====================

# 城市名称到adcode的映射（常用城市）
CITY_ADCODE = {
    "北京": "110000", "上海": "310000", "广州": "440100", "深圳": "440300",
    "杭州": "330100", "成都": "510100", "武汉": "420100", "西安": "610100",
    "南京": "320100", "重庆": "500000", "天津": "120000", "苏州": "320500",
    "郑州": "410100", "长沙": "430100", "青岛": "370200", "大连": "210200",
    "厦门": "350200", "宁波": "330200", "无锡": "320200", "佛山": "440600",
    "东莞": "441900", "合肥": "340100", "昆明": "530100", "福州": "350100",
    "济南": "370100", "石家庄": "130100", "沈阳": "210100", "长春": "220100",
    "哈尔滨": "230100", "南昌": "360100", "太原": "140100", "南宁": "450100",
    "贵阳": "520100", "兰州": "620100", "海口": "460100", "三亚": "460200",
    "呼和浩特": "150100", "银川": "640100", "西宁": "630100", "拉萨": "540100",
    "乌鲁木齐": "650100", "台北": "710000", "香港": "810000", "澳门": "820000"
}

async def get_city_adcode(city_name: str, amap_key: str) -> str:
    """
    通过高德地理编码API获取城市adcode
    
    参数:
        city_name: 城市名称
        amap_key: 高德API密钥
    
    返回:
        城市adcode，失败返回None
    """
    # 先检查常用城市映射表
    for city, adcode in CITY_ADCODE.items():
        if city in city_name:
            return adcode
    
    # 不在映射表中，通过API查询
    try:
        async with httpx.AsyncClient() as client:
            geo_url = f"https://restapi.amap.com/v3/geocode/geo"
            params = {
                "address": city_name,
                "key": amap_key
            }
            response = await client.get(geo_url, params=params)
            data = response.json()
            
            if data.get("status") == "1" and data.get("geocodes"):
                return data["geocodes"][0]["adcode"]
            return None
    except Exception as e:
        print(f"获取城市adcode失败: {e}")
        return None

async def get_weather(city: str, units: str = "metric", extensions: str = "base") -> str:
    """
    获取天气信息（使用高德天气API）
    
    参数:
        city: 城市名称（如：北京、上海）
        units: 单位（保留兼容性，高德API默认使用摄氏度）
        extensions: 气象类型，"base"=实况天气，"all"=预报天气
    
    返回:
        格式化的天气信息字符串
    """
    amap_key = os.environ.get("AMAP_API_KEY")
    
    if not amap_key:
        return "错误：未设置AMAP_API_KEY环境变量\n请在.env文件中添加：AMAP_API_KEY=your_key"
    
    try:
        # 1. 获取城市adcode
        adcode = await get_city_adcode(city, amap_key)
        
        if not adcode:
            return f"未找到城市 '{city}' 的地理信息"
        
        # 2. 调用高德天气API
        async with httpx.AsyncClient() as client:
            weather_url = "https://restapi.amap.com/v3/weather/weatherInfo"
            params = {
                "key": amap_key,
                "city": adcode,
                "extensions": extensions,
                "output": "json"
            }
            
            response = await client.get(weather_url, params=params)
            data = response.json()
            
            if data.get("status") != "1":
                return f"天气查询失败: {data.get('info', '未知错误')}"
            
            # 3. 解析并格式化返回数据
            if extensions == "base":
                # 实况天气
                lives = data.get("lives", [])
                if lives:
                    live = lives[0]
                    result = (
                        f"🌍 {live.get('province', '')}{live.get('city', '')} 当前天气：\n"
                        f"🌡️ 温度：{live.get('temperature', '')}°C\n"
                        f"💧 湿度：{live.get('humidity', '')}%\n"
                        f"🌤️ 天气：{live.get('weather', '')}\n"
                        f"🌬️ 风向：{live.get('winddirection', '')}\n"
                        f"💨 风力：{live.get('windpower', '')}级\n"
                        f"📅 更新时间：{live.get('reporttime', '')}"
                    )
                    return result
            
            elif extensions == "all":
                # 预报天气
                forecasts = data.get("forecasts", [])
                if forecasts:
                    forecast = forecasts[0]
                    city_info = f"{forecast.get('province', '')}{forecast.get('city', '')}"
                    casts = forecast.get("casts", [])
                    
                    result = f"🌍 {city_info} 天气预报：\n"
                    result += f"📅 发布时间：{forecast.get('reporttime', '')}\n\n"
                    
                    for cast in casts[:4]:  # 显示前4天预报
                        result += (
                            f"📆 {cast.get('date', '')} ({cast.get('week', '')})\n"
                            f"   白天：{cast.get('dayweather', '')} {cast.get('daytemp', '')}°C "
                            f"{cast.get('daywind', '')} {cast.get('daypower', '')}级\n"
                            f"   晚上：{cast.get('nightweather', '')} {cast.get('nighttemp', '')}°C "
                            f"{cast.get('nightwind', '')} {cast.get('nightpower', '')}级\n\n"
                        )
                    
                    return result
            
            return "未获取到天气数据"
            
    except Exception as e:
        return f"获取天气失败: {e}"

# ==================== Tavily搜索模块 ====================

class TavilySearch:
    """Tavily搜索封装"""
    
    def __init__(self):
        self.api_key = os.environ.get("TAVILY_API_KEY", "")
    
    async def search(self, query: str, max_results: int = 5) -> str:
        """
        使用Tavily API进行搜索
        
        参数:
            query: 搜索查询
            max_results: 最大结果数
        """
        if not self.api_key:
            return "错误：请设置TAVILY_API_KEY环境变量"
        
        try:
            url = "https://api.tavily.com/search"
            payload = {
                "api_key": self.api_key,
                "query": query,
                "max_results": max_results,
                "search_depth": "basic",
                "include_answer": True,
                "include_raw_content": False
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(url, json=payload, timeout=30.0)
                data = response.json()
                
                if data.get("answer"):
                    result = f"🔍 搜索结果摘要：\n{data['answer']}\n\n"
                else:
                    result = f"🔍 搜索结果：\n"
                
                # 添加具体结果
                if "results" in data:
                    for i, item in enumerate(data["results"][:max_results], 1):
                        result += f"\n{i}. {item['title']}\n"
                        result += f"   {item['url']}\n"
                        if item.get("content"):
                            result += f"   {item['content'][:200]}...\n"
                
                return result
        except Exception as e:
            return f"Tavily搜索失败: {e}"

# 全局实例
tavily = TavilySearch()

async def combined_query(query: str, city: str = "东莞", weather_days: int = 7, 
                      top_k: int = 3, search_results: int = 3) -> str:
    """
    组合查询：同时调用向量数据库、Tavily搜索和天气查询
    
    参数:
        query: 用户查询问题
        city: 天气查询城市，默认东莞
        weather_days: 天气预报天数，默认7天
        top_k: 向量数据库返回文档数
        search_results: Tavily搜索结果数
    
    返回:
        综合查询结果
    """
    import asyncio
    
    # 并行执行三个查询
    tasks = []
    
    # 1. 向量数据库搜索
    async def search_vector():
        try:
            answer, docs = vector_db.search(query, top_k)
            return f"📚 知识库回答：\n{answer}\n\n【参考文档数：{len(docs)}】"
        except Exception as e:
            return f"❌ 知识库搜索失败：{e}"
    tasks.append(asyncio.create_task(search_vector()))
    
    # 2. Tavily搜索
    async def search_tavily():
        try:
            return await tavily.search(query, max_results=search_results)
        except Exception as e:
            return f"❌ Tavily搜索失败：{e}"
    tasks.append(asyncio.create_task(search_tavily()))
    
    # 3. 天气查询
    async def get_weather_info():
        try:
            # 高德API只支持4天预报，使用all获取所有可用预报
            return await get_weather(city, extensions="all")
        except Exception as e:
            return f"❌ 天气查询失败：{e}"
    tasks.append(asyncio.create_task(get_weather_info()))
    
    # 等待所有任务完成
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # 组合结果
    final_result = f"{'='*60}\n"
    final_result += f"🌾 农业智能助手综合查询\n"
    final_result += f"{'='*60}\n\n"
    
    final_result += f"📝 用户问题：{query}\n"
    final_result += f"🌍 查询城市：{city}\n\n"
    
    final_result += f"{'-'*60}\n"
    final_result += f"{results[0]}\n\n"
    
    final_result += f"{'-'*60}\n"
    final_result += f"{results[1]}\n\n"
    
    final_result += f"{'-'*60}\n"
    final_result += f"{results[2]}\n"
    
    final_result += f"{'='*60}\n"
    
    return final_result

# ==================== MCP工具定义 ====================

@app.list_tools()
async def list_tools() -> list[Tool]:
    """列出所有可用工具"""
    return [
        # 组合查询工具（新）
        Tool(
            name="combined_query",
            description="农业智能助手综合查询：同时查询知识库、网络搜索和天气信息。适合获取全面的农业技术指导和实时天气信息。",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "用户的问题或查询内容"
                    },
                    "city": {
                        "type": "string",
                        "description": "天气查询城市，例如：'东莞'、'北京'",
                        "default": "东莞"
                    },
                    "weather_days": {
                        "type": "number",
                        "description": "天气预报天数（高德API最多支持4天）",
                        "default": 4,
                        "minimum": 1,
                        "maximum": 4
                    },
                    "top_k": {
                        "type": "number",
                        "description": "向量数据库返回相关文档数量",
                        "default": 3,
                        "minimum": 1,
                        "maximum": 10
                    },
                    "search_results": {
                        "type": "number",
                        "description": "Tavily搜索结果数量",
                        "default": 3,
                        "minimum": 1,
                        "maximum": 10
                    }
                },
                "required": ["query"]
            }
        ),
        
        # 向量数据库搜索工具
        Tool(
            name="search_vector_db",
            description="在向量数据库中搜索相关文档并基于文档内容回答用户问题",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "用户的问题或查询内容"
                    },
                    "top_k": {
                        "type": "number",
                        "description": "返回相关文档数量",
                        "default": 3,
                        "minimum": 1,
                        "maximum": 10
                    }
                },
                "required": ["query"]
            }
        ),
        
        # 天气查询工具
        Tool(
            name="get_weather",
            description="获取指定城市的天气信息（使用高德天气API）",
            inputSchema={
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，例如：'北京'、'上海'"
                    },
                    "extensions": {
                        "type": "string",
                        "description": "气象类型：'base'(当前实况天气) 或 'all'(未来4天预报)",
                        "default": "base",
                        "enum": ["base", "all"]
                    }
                },
                "required": ["city"]
            }
        ),
        
        # Tavily搜索工具
        Tool(
            name="tavily_search",
            description="使用Tavily搜索引擎进行网络搜索，获取最新信息",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索查询内容"
                    },
                    "max_results": {
                        "type": "number",
                        "description": "返回结果数量",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 10
                    }
                },
                "required": ["query"]
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """处理工具调用"""
    try:
        if name == "combined_query":
            query = arguments.get("query", "")
            city = arguments.get("city", "东莞")
            weather_days = arguments.get("weather_days", 4)
            top_k = arguments.get("top_k", 3)
            search_results = arguments.get("search_results", 3)
            
            result = await combined_query(query, city, weather_days, top_k, search_results)
            return [TextContent(type="text", text=result)]
        
        elif name == "search_vector_db":
            query = arguments.get("query", "")
            top_k = arguments.get("top_k", 3)
            answer, docs = vector_db.search(query, top_k)
            
            result = f"📚 知识库搜索结果：\n\n{answer}\n\n"
            result += f"【参考文档数：{len(docs)}】"
            
            return [TextContent(type="text", text=result)]
        
        elif name == "get_weather":
            city = arguments.get("city", "")
            extensions = arguments.get("extensions", "base")
            
            weather_info = await get_weather(city, extensions=extensions)
            return [TextContent(type="text", text=weather_info)]
        
        elif name == "tavily_search":
            query = arguments.get("query", "")
            max_results = arguments.get("max_results", 5)
            
            search_results = await tavily.search(query, max_results)
            return [TextContent(type="text", text=search_results)]
        
        else:
            return [TextContent(type="text", text=f"未知工具：{name}")]
    
    except Exception as e:
        return [TextContent(type="text", text=f"工具执行失败: {e}")]

# ==================== MCP资源定义 ====================

@app.list_resources()
async def list_resources() -> list[Resource]:
    """列出可用资源"""
    return [
        Resource(
            uri="vector://db/info",
            name="向量数据库信息",
            description="向量数据库的连接状态和元信息",
            mimeType="text/plain"
        )
    ]

@app.read_resource()
async def read_resource(uri: str) -> str:
    """读取资源"""
    if uri == "vector://db/info":
        if vector_db._initialized:
            return f"向量数据库已连接\n集合名称: {vector_db.collection_name}\n状态: 正常"
        else:
            return "向量数据库未初始化\n请先调用搜索工具触发初始化"
    return "未知资源"

# ==================== 服务器启动 ====================

async def main():
    """启动MCP服务器"""
    from mcp.server.stdio import stdio_server
    
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
