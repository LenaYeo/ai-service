# config.py
import os
from pathlib import Path
from dotenv import load_dotenv

# 기본 경로 설정
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = os.path.join(BASE_DIR, "data")
PROMPTS_DIR = os.path.join(BASE_DIR, "prompts")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

# 필요한 디렉터리 생성
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PROMPTS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# API 키 설정
load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")
NEWS_API_KEY = os.environ.get("NEWS_API_KEY", "")  # 뉴스 API 키 (예: NewsAPI)

# 모델 설정
DEFAULT_MODEL = "gpt-4o-mini"
FAST_MODEL = "gpt-3.5-turbo"

# 분석 대상 기술 분야 및 키워드
TECH_COMPANIES = [
    "Google", "Alphabet", "DeepMind",
    "Microsoft", "OpenAI",
    "Meta", "Facebook",
    "Amazon", "AWS",
    "Apple",
    "NVIDIA",
    "IBM",
    "Tesla",
    "Baidu",
    "Tencent",
    "Samsung",
    "Intel",
    "AMD",
    "Oracle",
    "Anthropic"
]
