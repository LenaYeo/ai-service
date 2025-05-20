# prompts/news_prompts.py
from langchain_core.prompts import ChatPromptTemplate

def get_news_analysis_prompt():
    """뉴스와 IR 자료 분석 프롬프트"""
    return ChatPromptTemplate.from_template("""
    다음은 {tech_field} 분야의 뉴스 기사와 기업 IR 자료입니다:
    
    ## 뉴스 기사:
    {news_data}
    
    ## 기업 IR 자료:
    {ir_data}
    
    위 자료를 분석하여 다음 정보를 제공해주세요:
    
    1. {tech_field} 분야의 최신 연구 동향 요약 (주요 발전, 돌파구, 새로운 접근 방식)
    2. 기업들의 기술 개발 현황 및 투자 방향
    3. 주요 기업들의 핵심 프로젝트와 협업 현황
    4. 향후 6개월~1년 내 예상되는 기술 발전 방향
    
    각 정보는 항목별로 구분하여 작성하고, 가능한 구체적인 내용과 수치를 포함해 주세요.
    """)

def get_tech_trends_summary_prompt():
    """기술 동향 요약 프롬프트"""
    return ChatPromptTemplate.from_template("""
    다음은 {tech_field} 분야의 기술 뉴스 데이터입니다:
    
    {news_data}
    
    위 뉴스 데이터를 분석하여 {tech_field} 분야의 최신 연구 및 기술 동향을 요약해주세요.
    다음 항목을 포함하여 JSON 형식으로 응답해주세요:
    
    1. "key_developments": 주요 기술 발전 (목록 형태, 최대 5개)
    2. "main_players": 주요 기업 및 연구 기관 (목록 형태)
    3. "emerging_technologies": 떠오르는 기술 (목록 형태, 최대 3개)
    4. "challenges": 현재 이 분야가 직면한 과제 (목록 형태)
    5. "future_directions": 향후 발전 방향 (목록 형태)
    
    결과는 다음 형식과 같이 구조화된 JSON으로 제공해주세요:
    {{
      "key_developments": ["발전1", "발전2", ...],
      "main_players": ["기업/기관1", "기업/기관2", ...],
      "emerging_technologies": ["기술1", "기술2", ...],
      "challenges": ["과제1", "과제2", ...],
      "future_directions": ["방향1", "방향2", ...]
    }}
    """)

def get_ir_analysis_prompt():
    """기업 IR 자료 분석 프롬프트"""
    return ChatPromptTemplate.from_template("""
    다음은 {company_name}의 IR 자료입니다:
    
    {ir_data}
    
    위 IR 자료를 분석하여 다음 정보를 JSON 형식으로 추출해주세요:
    
    1. "company": 회사명
    2. "tech_investments": 기술 투자 분야 및 금액 (목록 형태)
    3. "rd_focus": R&D 중점 영역 (목록 형태)
    4. "collaborations": 주요 협업 기관 또는 기업 (목록 형태)
    5. "future_plans": 향후 기술 개발 계획 (목록 형태)
    6. "financial_metrics": 관련 재무 지표 (객체 형태)
    
    결과는 다음과 같이 구조화된 JSON으로 제공해주세요:
    {{
      "company": "회사명",
      "tech_investments": ["투자 분야1: 금액", "투자 분야2: 금액", ...],
      "rd_focus": ["중점 영역1", "중점 영역2", ...],
      "collaborations": ["협업 기관/기업1", "협업 기관/기업2", ...],
      "future_plans": ["계획1", "계획2", ...],
      "financial_metrics": {{
        "r_and_d_spending": "R&D 지출액",
        "revenue_from_new_tech": "신기술 매출 비중",
        "growth_rate": "성장률"
      }}
    }}
    """)