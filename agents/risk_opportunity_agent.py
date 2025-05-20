# agents/risk_opportunity_agent.py
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import Dict, List, Optional
import logging

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

from config import OPENAI_API_KEY, DEFAULT_MODEL
# from utils.data_utils import save_intermediate_result

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 프롬프트 정의
RISK_OPPORTUNITY_PROMPT_TEMPLATE = """
당신은 기술 전략 분석가입니다. 다음 정보를 종합하여 주요 기술 및 시장 동향과 관련된 리스크와 기회 요인을 분석해주세요:

1.  **핵심 기술 요약**:
    {tech_summary_str}

2.  **예측된 기술 트렌드**:
    {trend_prediction_str}

3.  **관련 정책 동향**:
    {policy_highlights}
    
4.  **뉴스 및 시장 반응**:
    {news_highlights}
    
5.  **이전 검증 단계에서 발견된 이슈 (있을 경우)**:
    {validation_issues_str}

각 리스크와 기회 요인에 대해 다음 항목을 포함하여 JSON 형식으로 답변해주세요:
-   `type`: "Risk" 또는 "Opportunity"
-   `factor_name`: 요인명
-   `description`: 상세 설명
-   `related_technologies`: 관련된 핵심 기술 (리스트)
-   `potential_magnitude`: 잠재적 영향력 (High, Medium, Low)
-   `timeframe`: 예상 발현 시기 (Short-term, Mid-term, Long-term)
-   `mitigation_or_leverage_strategy` (선택적): 리스크 완화 또는 기회 활용 방안 제안

예시:
{{
  "analysis_summary": "전반적인 리스크/기회 분석 요약...",
  "identified_factors": [
    {{
      "type": "Risk",
      "factor_name": "AI 기술의 규제 강화",
      "description": "각국 정부의 AI 윤리 및 데이터 보호 규제 강화로 인해 기술 개발 및 도입에 제약이 발생할 수 있음.",
      "related_technologies": ["Generative AI", "Explainable AI"],
      "potential_magnitude": "High",
      "timeframe": "Mid-term",
      "mitigation_or_leverage_strategy": "선제적인 윤리 가이드라인 수립 및 투명성 확보 노력 필요."
    }},
    {{
      "type": "Opportunity",
      "factor_name": "로봇 자동화 시장 확대",
      "description": "제조업 및 서비스업에서의 인력 부족 문제와 생산성 향상 요구로 로봇 자동화 솔루션 도입 가속화.",
      "related_technologies": ["Robotics", "Computer Vision", "Edge AI"],
      "potential_magnitude": "High",
      "timeframe": "Short-term",
      "mitigation_or_leverage_strategy": "특화된 산업별 로봇 솔루션 개발 및 시장 선점 전략."
    }}
  ]
}}
"""

class RiskOpportunityAgent:
    """리스크 및 기회 요인을 분석하는 에이전트"""
    def __init__(self, model_name=DEFAULT_MODEL):
        self.llm = ChatOpenAI(
            api_key=OPENAI_API_KEY,
            model=model_name,
            temperature=0.3
        )
        self.prompt = PromptTemplate.from_template(RISK_OPPORTUNITY_PROMPT_TEMPLATE)
        self.chain = self.prompt | self.llm | JsonOutputParser()

    def _format_input_for_prompt(self, data: Optional[Dict], name: str) -> str:
        if not data:
            return f"{name} 정보 없음."
        # 간단히 JSON 문자열로 변환하거나, 주요 내용만 추출
        import json
        return json.dumps(data, indent=2, ensure_ascii=False)[:1500] # 길이 제한

    def _prepare_input_data(self, data: Dict, max_items=3, max_length=200) -> str:
        """LLM 입력용 하이라이트 문자열 생성"""
        if not data:
            return "데이터 없음"
        highlights = []
        if "policies" in data: # policy_data
            for policy in data.get("policies", [])[:max_items]:
                highlights.append(f"- {policy.get('title', '')} ({policy.get('region')}): {policy.get('summary', '')[:max_length]}...")
        elif "news_items" in data: # news_data (tech_news part)
             for item in data.get("news_items", [])[:max_items]:
                highlights.append(f"- {item.get('title', '')}: {item.get('content', '')[:max_length]}...")
        
        if not highlights:
            return "유의미한 항목 없음"
        return "\n".join(highlights)

    def run(self, research_data: Dict, news_data: Dict, policy_data: Dict,
            tech_summary: Dict, trend_prediction: Dict, 
            validation_errors: Optional[List[str]] = None) -> Dict:
        logger.info("리스크 및 기회 분석 시작...")

        tech_summary_str = self._format_input_for_prompt(tech_summary, "핵심 기술 요약")
        trend_prediction_str = self._format_input_for_prompt(trend_prediction, "예측된 기술 트렌드")
        
        policy_highlights = self._prepare_input_data(policy_data)

        # news_data에서 tech_news 부분만 사용
        tech_news_content = {}
        if news_data and 'tech_news_analysis' in news_data:
            all_tech_news_items = []
            for _keyword, kw_data in news_data['tech_news_analysis'].items():
                if 'news_items' in kw_data:
                    all_tech_news_items.extend(kw_data['news_items'])
            tech_news_content = {"news_items": all_tech_news_items}
        news_highlights = self._prepare_input_data(tech_news_content)
        
        validation_issues_str = "없음"
        if validation_errors:
            validation_issues_str = "\n- ".join(validation_errors)
            logger.info(f"이전 검증 오류 반영: {validation_issues_str}")


        try:
            response = self.chain.invoke({
                "tech_summary_str": tech_summary_str,
                "trend_prediction_str": trend_prediction_str,
                "policy_highlights": policy_highlights,
                "news_highlights": news_highlights,
                "validation_issues_str": validation_issues_str
            })
            logger.info("리스크 및 기회 분석 완료.")
            # save_intermediate_result(response, "risk_opportunity_agent_output") # 필요시 저장
            return response
        except Exception as e:
            logger.error(f"리스크 및 기회 분석 중 오류 발생: {e}")
            return {
                "analysis_summary": "오류로 인해 분석에 실패했습니다.",
                "identified_factors": [{"type": "Error", "factor_name": "분석 실패", "description": str(e)}]
            }

# if __name__ == '__main__':
#     agent = RiskOpportunityAgent()
#     dummy_summary = {"overall_summary": "Key tech summary."}
#     dummy_trend = {"overall_trend_score": 70}
#     dummy_policy = {"policies": [{"title": "Policy C", "summary": "Summary of C..."}]}
#     dummy_news = {"tech_news_analysis": {"AI": {"news_items": [{"title": "News B", "content": "Content of B..."}]}}}
#     dummy_research = {} # 이 에이전트는 research_data를 직접 사용하지 않음 (요약을 통해 간접 사용)
    
#     analysis = agent.run(dummy_research, dummy_news, dummy_policy, dummy_summary, dummy_trend, validation_errors=["이전 예측이 너무 낙관적임"])
#     print(analysis)
