"""
农业智能问答 Agent
根据用户问题自动选择合适的工具并综合回答
"""
import asyncio
import os
import re

# ==================== 加载环境变量 ====================
from dotenv import load_dotenv

# 获取当前脚本所在目录的 .env 文件
script_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(script_dir, '.env')

# 加载 .env 文件（明确指定路径）
load_dotenv(env_path)

# ==================== API 密钥配置 ====================
# 从环境变量读取 API 密钥（必须从 .env 加载，不要使用硬编码）
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
AMAP_API_KEY = os.environ.get("AMAP_API_KEY")
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_API_BASE = os.environ.get("OPENAI_API_BASE") or "https://dashscope.aliyuncs.com/compatible-mode/v1"

# 验证关键配置
if not DASHSCOPE_API_KEY:
    print("⚠️  警告: DASHSCOPE_API_KEY 未设置")
if not TAVILY_API_KEY:
    print("⚠️  警告: TAVILY_API_KEY 未设置")
if not AMAP_API_KEY:
    print("⚠️  警告: AMAP_API_KEY 未设置")
if not OPENAI_API_KEY:
    print("⚠️  警告: OPENAI_API_KEY 未设置")

# 设置环境变量（仅当值存在时才设置，用于 LangChain 组件）
if DASHSCOPE_API_KEY:
    os.environ['DASHSCOPE_API_KEY'] = DASHSCOPE_API_KEY
if TAVILY_API_KEY:
    os.environ['TAVILY_API_KEY'] = TAVILY_API_KEY
if AMAP_API_KEY:
    os.environ['AMAP_API_KEY'] = AMAP_API_KEY
if OPENAI_API_KEY:
    os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
if OPENAI_API_BASE:
    os.environ['OPENAI_API_BASE'] = OPENAI_API_BASE

import httpx

# 尝试导入 LangChain 组件
try:
    from langchain_community.embeddings import DashScopeEmbeddings
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import PromptTemplate
    from pymilvus import connections, utility, Collection
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  警告: LangChain 组件导入失败")
    print(f"  错误: {e}")
    print(f"  请运行: pip install langchain-community langchain-openai langchain-core pymilvus")
    LANGCHAIN_AVAILABLE = False

# ==================== 工具类定义 ====================

class VectorDBTool:
    """向量数据库工具"""
    
    def __init__(self):
        self.collection_name = "agriculture_kb"
        self.embeddings = None
        self.collection = None
        self._initialized = False
    
    def initialize(self):
        if self._initialized:
            return True
        
        if not LANGCHAIN_AVAILABLE:
            print("❌ LangChain 组件不可用，知识库功能禁用")
            return False
            
        try:
            self.embeddings = DashScopeEmbeddings(model="text-embedding-v2")
            connections.connect(host="localhost", port="19530")
            
            if utility.has_collection(self.collection_name):
                self.collection = Collection(name=self.collection_name)
                self.collection.load()
                self._initialized = True
                print(f"✅ 知识库连接成功")
                return True
            else:
                print(f"⚠️  集合不存在: {self.collection_name}")
                return False
        except Exception as e:
            print(f"❌ 知识库连接失败: {e}")
            return False
    
    def search(self, query: str, top_k: int = 3) -> str:
        """搜索知识库"""
        if not self._initialized:
            if not self.initialize():
                return "知识库未初始化"
        
        try:
            query_vector = self.embeddings.embed_query(query)
            search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
            results = self.collection.search(
                data=[query_vector],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                output_fields=["text"]
            )
            
            context = "\n\n".join([
                f"【文档 {i+1}】{result.entity.get('text', '')}"
                for i, result in enumerate(results[0])
            ])
            return context
        except Exception as e:
            return f"搜索失败: {e}"

class WeatherTool:
    """天气查询工具"""
    
    def __init__(self):
        self.city_adcodes = {
            "东莞": "441900", "北京": "110000", "上海": "310000",
            "广州": "440100", "深圳": "440300", "杭州": "330100",
            "成都": "510100", "武汉": "420100", "西安": "610100",
            "重庆": "500000", "天津": "120000", "苏州": "320500"
        }
    
    async def query(self, city: str) -> str:
        """查询天气"""
        if not AMAP_API_KEY:
            return "未配置天气API"
        
        try:
            adcode = self.city_adcodes.get(city, "441900")
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "https://restapi.amap.com/v3/weather/weatherInfo",
                    params={
                        "key": AMAP_API_KEY,
                        "city": adcode,
                        "extensions": "all",
                        "output": "json"
                    }
                )
                data = response.json()
                
                if data.get("status") == "1" and data.get("forecasts"):
                    forecast = data["forecasts"][0]
                    casts = forecast.get("casts", [])
                    
                    result = f"{forecast.get('province', '')}{forecast.get('city', '')}天气预报:\n"
                    for cast in casts[:4]:
                        result += f"{cast.get('date', '')}: "
                        result += f"白天{cast.get('daytemp', '')}°C {cast.get('dayweather', '')}, "
                        result += f"夜间{cast.get('nighttemp', '')}°C {cast.get('nightweather', '')}\n"
                    
                    return result
                
                return "天气查询失败"
        except Exception as e:
            return f"天气查询失败: {e}"

class TavilyTool:
    """Tavily搜索工具"""
    
    async def search(self, query: str, max_results: int = 3) -> str:
        """网络搜索"""
        if not TAVILY_API_KEY:
            return "未配置搜索API"
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.tavily.com/search",
                    json={
                        "api_key": TAVILY_API_KEY,
                        "query": query,
                        "max_results": max_results,
                        "search_depth": "basic"
                    },
                    timeout=30.0
                )
                data = response.json()
                
                results = []
                if data.get("answer"):
                    results.append(data["answer"])
                
                if "results" in data:
                    for item in data["results"][:max_results]:
                        results.append(f"- {item.get('title', '')}: {item.get('content', '')[:150]}")
                
                return "\n".join(results) if results else "无搜索结果"
        except Exception as e:
            return f"搜索失败: {e}"

# ==================== Agent 定义 ====================

class AgricultureAgent:
    """农业智能问答 Agent"""
    
    def __init__(self):
        self.vector_db = VectorDBTool()
        self.weather = WeatherTool()
        self.tavily = TavilyTool()
        
        if LANGCHAIN_AVAILABLE:
            self.llm = ChatOpenAI(
                model="qwen-plus",
                temperature=0.3,
                max_tokens=1500
            )
        else:
            self.llm = None
        
        self._tools_initialized = False
    
    async def initialize(self):
        """初始化所有工具"""
        if self._tools_initialized:
            return
        
        print("正在初始化农业智能 Agent...")
        
        if LANGCHAIN_AVAILABLE:
            self.vector_db.initialize()
            print("✅ Agent 初始化完成\n")
        else:
            print("⚠️  LangChain 不可用，仅启用天气和网络搜索\n")
        
        self._tools_initialized = True
    
    def detect_intent(self, query: str) -> dict:
        """检测用户意图"""
        query_lower = query.lower()
        
        intents = {
            "knowledge": False,
            "weather": False,
            "search": False,
            "cities": []
        }
        
        # 检测天气相关
        weather_keywords = ["天气", "气温", "温度", "下雨", "晴天", "阴天", "预报", "雨雪", "寒潮", "高温"]
        if any(kw in query_lower for kw in weather_keywords):
            intents["weather"] = True
        
        # 提取城市名
        cities = ["东莞", "北京", "上海", "广州", "深圳", "杭州", "成都", "武汉", "西安", "重庆", "天津", "苏州"]
        for city in cities:
            if city in query:
                intents["cities"].append(city)
        
        # 如果没有城市但问天气，默认东莞
        if intents["weather"] and not intents["cities"]:
            intents["cities"] = ["东莞"]
        
        # 农业技术问题默认使用知识库
        agri_keywords = ["作物", "防冻", "病虫害", "施肥", "灌溉", "种植", "技术", "措施", "指导"]
        if any(kw in query_lower for kw in agri_keywords):
            intents["knowledge"] = True
        
        # 如果没有特定意图，默认使用知识库+搜索
        if not any(intents.values()):
            intents["knowledge"] = True
            intents["search"] = True
        
        return intents
    
    async def process(self, query: str) -> str:
        """处理用户查询"""
        if not self._tools_initialized:
            await self.initialize()
        
        # 检测意图
        intents = self.detect_intent(query)
        
        print(f"检测到的意图: {intents}\n")
        
        # 并行执行相关工具
        tasks = []
        
        if intents["knowledge"]:
            async def search_kb():
                print("🔍 正在搜索农业知识库...")
                return self.vector_db.search(query)
            tasks.append(asyncio.create_task(search_kb()))
        
        if intents["search"]:
            async def search_web():
                print("🌐 正在搜索网络...")
                return await self.tavily.search(query)
            tasks.append(asyncio.create_task(search_web()))
        
        if intents["weather"]:
            city = intents["cities"][0] if intents["cities"] else "东莞"
            async def get_weather():
                print(f"🌤️ 正在查询{city}天气...")
                return await self.weather.query(city)
            tasks.append(asyncio.create_task(get_weather()))
        
        # 执行所有任务
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 构建回答
        response = self._build_response(query, intents, results)
        return response
    
    def _build_response(self, query: str, intents: dict, results: list) -> str:
        """构建最终回答"""
        if not LANGCHAIN_AVAILABLE:
            # LangChain 不可用，简单拼接结果
            context_parts = []
            
            if intents["weather"] and len(results) > 0:
                context_parts.append(f"【天气预报】\n{results[0]}")
            
            if intents["search"] and len(results) > 1:
                context_parts.append(f"【网络搜索结果】\n{results[1]}")
            
            if context_parts:
                return "\n\n".join(context_parts)
            else:
                return "抱歉，当前服务不可用，请检查依赖安装。"
        
        # 汇总所有工具结果
        context_parts = []
        
        if intents["knowledge"] and len(results) > 0:
            context_parts.append(f"【农业知识库结果】\n{results[0]}")
        
        if intents["search"] and len(results) > 1:
            context_parts.append(f"【网络搜索结果】\n{results[1]}")
        
        if intents["weather"] and len(results) > 2:
            context_parts.append(f"【天气预报】\n{results[2]}")
        
        context = "\n\n".join(context_parts)
        
        # 使用 LLM 生成最终回答
        prompt_template = PromptTemplate(
            input_variables=["question", "context"],
            template="""你是一个专业的农业智能助手，请基于以下信息回答用户的问题。
请严格按照以下流程：向量数据库检索—>若信息不够完整—>通过Tavily搜索补充信息—>再整合信息思考一遍合理性和真实性->给出答案。
                
可用文档片段：  片段1: [农业技术手册-小麦篇] 小麦赤霉病防治关键期为“齐穗至扬花初期”。一旦错过此窗口期，后期打药效果极差。 
片段2: [植保站2024年4月预警] 建议选用氰烯菌酯、戊唑醇、咪鲜胺等高效低毒药剂。对于抗性较强地区，推荐使用丙硫菌唑。 
片段3: [农药安全使用规范] 使用戊唑醇时，每亩用量不应超过30毫升，且每个生长季最多使用2次，以防止产生药害或抗药性。 

用户问题：我家小麦马上要扬花抽穗了，预防赤霉病该什么时候打药？可以用戊唑醇吗？有什么注意事项？

用户问题：{question}

可用信息：
{context}

请遵循以下原则：
1. 综合所有提供的信息，给出准确、有用的回答
2. 如果知识库中有技术指导，优先使用并详细说明
3. 如果网络搜索有补充信息，可以适当引用
4. 如果包含天气信息，可以结合农事建议
5. 回答要简洁明了，结构清晰
6. 使用专业的农业术语
7. 如果某些信息缺失，明确说明

请回答："""
        )
        
        prompt_text = prompt_template.format(context=context, question=query)
        response = self.llm.invoke(prompt_text)
        
        return response.content

# ==================== 交互式问答 ====================

async def interactive_agent():
    """交互式问答模式"""
    print("=" * 70)
    print("🌾 农业智能问答 Agent")
    print("=" * 70)
    print("\n我可以帮您：")
    print("  📚 查询农业技术指导（作物防冻、病虫害防治、种植技术等）")
    print("  🌤️ 查询天气信息（支持东莞、北京、上海等城市）")
    print("  🌐 搜索最新农业资讯")
    print("\n输入您的问题，我会自动判断并调用合适的工具")
    print("输入 'quit' 或 'exit' 退出")
    print("-" * 70 + "\n")
    
    agent = AgricultureAgent()
    await agent.initialize()
    
    while True:
        try:
            user_input = input("\n👤 您: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q', '退出']:
                print("\n👋 感谢使用，再见！")
                break
            
            if not user_input:
                print("⚠️  请输入有效的问题")
                continue
            
            print("\n🤖 正在思考...\n")
            response = await agent.process(user_input)
            
            print("\n" + "=" * 70)
            print("🤖 回答:")
            print("=" * 70)
            print(response)
            print()
            
        except KeyboardInterrupt:
            print("\n\n👋 感谢使用，再见！")
            break
        except Exception as e:
            print(f"\n❌ 处理出错: {e}")
            import traceback
            traceback.print_exc()

async def single_query(query: str):
    """单次查询"""
    print("=" * 70)
    print("🌾 农业智能问答 Agent")
    print("=" * 70)
    
    agent = AgricultureAgent()
    await agent.initialize()
    
    print(f"👤 您: {query}\n")
    print("🤖 正在思考...\n")
    
    response = await agent.process(query)
    
    print("=" * 70)
    print("🤖 回答:")
    print("=" * 70)
    print(response)

# ==================== 主函数 ====================

async def main():
    """主函数"""
    import sys
    args = sys.argv[1:]
    
    if "--interactive" in args:
        await interactive_agent()
    elif len(args) > 0 and not args[0].startswith("--"):
        # 第一个参数是问题
        query = " ".join(args)
        await single_query(query)
    else:
        # 显示使用说明
        print("\n🌾 农业智能问答 Agent - 使用说明")
        print("=" * 70)
        print("\n运行方式：")
        print("  python agri_agent.py \"您的问题\"          # 单次查询")
        print("  python agri_agent.py --interactive          # 交互式问答")
        print("\n示例：")
        print("  python agri_agent.py \"东莞明天的天气怎么样？\"")
        print("  python agri_agent.py \"冬小麦如何防冻？\"")
        print("  python agri_agent.py \"最新的农业技术有哪些？\"")

if __name__ == "__main__":
    asyncio.run(main())
