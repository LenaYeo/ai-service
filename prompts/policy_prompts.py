from langchain_core.prompts import ChatPromptTemplate

def get_policy_analysis_prompt():
    """간소화된 정책 분석용 프롬프트 템플릿"""
    template = """
    당신은 기술 정책 분석 전문가입니다. 다음 정책 데이터를 분석하여 JSON 형태로 요약해주세요.
    
    정책 데이터:
    {policy_data}
    
    다음 JSON 형식으로 응답해주세요:
    ```json
    {{
        "key_technologies": ["핵심 기술 1", "핵심 기술 2", "핵심 기술 3"],
        "policy_focus": ["정책 중점 분야 1", "정책 중점 분야 2"],
        "budget_info": {{
            "mentioned": true/false,
            "details": "언급된 예산 정보",
            "amount_scale": "대규모/중규모/소규모/언급 없음",
            "timeline": "예산 적용 기간"
        }},
        "timeframe": "정책 시행 기간/일정",
        "regulatory_aspects": ["규제 관련 측면 1", "규제 관련 측면 2"],
        "stakeholders": ["이해관계자 1", "이해관계자 2"],
        "international_context": "국제적 맥락 또는 협력 관련 정보",
        "potential_impact": "산업 및 기술 발전에 미치는 잠재적 영향"
    }}
    ```
    
    정책 데이터에서 확인할 수 없는 항목은 "데이터 없음"으로 표시하세요. 특히 예산 정보에 주의하여 금액, 증감률, 투자 계획 등을 자세히 분석해주세요. 전체 응답은 JSON 형식이어야 합니다.
    """
    return ChatPromptTemplate.from_template(template)
