# agents/validation_agent.py
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import Dict, List
import logging

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

from config import OPENAI_API_KEY, DEFAULT_MODEL

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 프롬프트 정의
VALIDATION_PROMPT_TEMPLATE = """
당신은 분석 결과 검증 전문가입니다. 다음 분석 결과들의 일관성, 논리적 타당성, 근거의 명확성을 검토해주세요.

1.  **핵심 기술 요약**:
    {tech_summary_str}

2.  **예측된 기술 트렌드**:
    {trend_prediction_str}

3.  **리스크 및 기회 분석**:
    {risk_opportunity_str}

4.  **원본 데이터 하이라이트 (참고용)**:
    - 연구: {research_highlights}
    - 뉴스: {news_highlights}
    - 정책: {policy_highlights}

검증 결과, 발견된 주요 이슈(문제점, 불일치, 근거 부족 등)들을 리스트로 제시하고, 전반적인 분석 결과가 유효한지 (is_valid: true/false) JSON 형식으로 답변해주세요.
이슈가 없다면 "issues" 리스트는 비워주세요.

예시:
{{
  "is_valid": false,
  "issues": [
    "특정 기술(예: '양자내성암호')의 트렌드 예측(급성장)이 핵심 기술 요약(언급 없음)과 불일치합니다.",
    "리스크 요인 중 '공급망 불안정'에 대한 근거 데이터가 부족합니다.",
    "기회 요인으로 제시된 '신시장 개척'이 너무 일반적이며 구체적인 기술 연관성이 부족합니다."
  ],
  "validation_summary": "몇 가지 불일치와 근거 부족 문제가 발견되어 추가 검토 및 수정이 필요합니다."
}}

또는 이슈가 없는 경우:
{{
  "is_valid": true,
  "issues": [],
  "validation_summary": "분석 결과가 전반적으로 일관되고 논리적 근거를 갖추고 있어 유효합니다."
}}
"""

class ValidationAgent:
    """분석 결과를 검증하는 에이전트 (또는 노드 함수)"""
    def __init__(self, model_name=DEFAULT_MODEL):
        self.llm = ChatOpenAI(
            api_key=OPENAI_API_KEY,
            model=model_name,
            temperature=0.1
        )
        self.prompt = PromptTemplate.from_template(VALIDATION_PROMPT_TEMPLATE)
        self.chain = self.prompt | self.llm | JsonOutputParser()

    def _format_input_for_prompt(self, data: Dict, name: str) -> str:
        if not data:
            return f"{name} 정보 없음."
        import json
        return json.dumps(data, indent=2, ensure_ascii=False)[:1000] # 길이 제한

    def _prepare_data_highlights(self, data: Dict, max_items=2, max_length=150) -> str:
        """LLM 입력용 데이터 하이라이트 문자열 생성"""
        if not data: return "데이터 없음"
        highlights = []
        if "papers" in data:
            for paper in data.get("papers", [])[:max_items]:
                highlights.append(f"- 연구: {paper.get('title', '')[:max_length]}...")
        elif "news_items" in data: # tech_news_content
             for item in data.get("news_items", [])[:max_items]:
                highlights.append(f"- 뉴스: {item.get('title', '')[:max_length]}...")
        elif "policies" in data:
            for policy in data.get("policies", [])[:max_items]:
                highlights.append(f"- 정책: {policy.get('title', '')[:max_length]}...")
        return "\n".join(highlights) if highlights else "하이라이트 없음"

    def run(self, research_data: Dict, news_data: Dict, policy_data: Dict,
            tech_summary: Dict, trend_prediction: Dict, risk_opportunity: Dict) -> Dict:
        logger.info("분석 결과 검증 시작...")

        tech_summary_str = self._format_input_for_prompt(tech_summary, "핵심 기술 요약")
        trend_prediction_str = self._format_input_for_prompt(trend_prediction, "예측된 기술 트렌드")
        risk_opportunity_str = self._format_input_for_prompt(risk_opportunity, "리스크 및 기회 분석")

        research_highlights = self._prepare_data_highlights(research_data)
        
        tech_news_content = {}
        if news_data and 'tech_news_analysis' in news_data:
            all_tech_news_items = []
            for _keyword, kw_data in news_data['tech_news_analysis'].items():
                if 'news_items' in kw_data:
                    all_tech_news_items.extend(kw_data['news_items'])
            tech_news_content = {"news_items": all_tech_news_items}
        news_highlights = self._prepare_data_highlights(tech_news_content)
        policy_highlights = self._prepare_data_highlights(policy_data)

        try:
            response = self.chain.invoke({
                "tech_summary_str": tech_summary_str,
                "trend_prediction_str": trend_prediction_str,
                "risk_opportunity_str": risk_opportunity_str,
                "research_highlights": research_highlights,
                "news_highlights": news_highlights,
                "policy_highlights": policy_highlights,
            })
            logger.info("분석 결과 검증 완료.")
            return response
        except Exception as e:
            logger.error(f"분석 결과 검증 중 오류 발생: {e}")
            return {
                "is_valid": False,
                "issues": [f"검증 과정에서 시스템 오류 발생: {str(e)}"],
                "validation_summary": "오류로 인해 검증에 실패했습니다."
            }

# if __name__ == '__main__':
#     agent = ValidationAgent()
#     # Dummy data for testing
#     dummy_summary = {"key_technologies": [{"name": "Tech Alpha"}], "overall_summary": "Summary Alpha"}
#     dummy_trend = {"predicted_trends": [{"technology_name": "Tech Alpha", "short_term_trend": "급성장"}]}
#     dummy_risk = {"identified_factors": [{"type": "Opportunity", "factor_name": "Market Growth for Alpha"}]}
#     dummy_research = {"papers": [{"title": "Alpha Research"}]}
#     dummy_news = {"tech_news_analysis": {"Alpha": {"news_items": [{"title": "Alpha News"}]}}}
#     dummy_policy = {"policies": [{"title": "Alpha Policy"}]}

#     validation = agent.run(dummy_research, dummy_news, dummy_policy, dummy_summary, dummy_trend, dummy_risk)
#     print(validation)
