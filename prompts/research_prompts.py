# prompts/research_prompts.py
from langchain_core.prompts import ChatPromptTemplate

def get_trend_analysis_prompt():
    """트렌드 분석 프롬프트 생성"""
    return ChatPromptTemplate.from_template("""
    다음은 로보틱스 및 AI 관련 연구 논문과 특허 데이터입니다:
    
    ## 연구 논문 초록:
    {paper_abstracts}
    
    위 자료를 종합적으로 분석하여 다음 정보를 JSON 형식으로 제공해주세요:
    1. 주요 연구 주제 (5개)
    2. 현재 및 미래 기술 트렌드 (3-5개)
    3. 주요 응용 분야 (3-5개)
        """)

def get_simple_analysis_prompt():
    """간단한 텍스트 기반 분석 프롬프트 생성"""
    return ChatPromptTemplate.from_template("""
    다음은 로보틱스 및 AI 관련 연구 논문과 특허 데이터입니다:
    
    ## 연구 논문 초록:
    {paper_abstracts}
    
    ## 관련 특허 데이터:
    {patent_data}
    
    위 자료를 종합적으로 분석하여 다음 정보를 제공해주세요:
    
    # 주요 연구 주제 (5개):
    1. 
    2. 
    3. 
    4. 
    5. 
    
    # 현재 및 미래 기술 트렌드 (3-5개):
    1. 
    2. 
    3. 
    4. 
    5. 
    
    # 주요 응용 분야 (3-5개):
    1. 
    2. 
    3. 
    4. 
    5. 
    """)