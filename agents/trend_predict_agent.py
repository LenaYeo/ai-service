# agents/trend_predict_agent.py
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import Dict, List, Any
import logging
import json

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

from config import OPENAI_API_KEY, DEFAULT_MODEL

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TREND_PREDICTION_PROMPT_TEMPLATE = """
당신은 저명한 미래 기술 트렌드 분석가입니다. 다음 정보를 바탕으로 주요 기술들의 향후 1-3년 간 트렌드를 예측해주세요.
특히, 제공된 **정량적 데이터(논문 수, 뉴스 언급 빈도, 기업 투자 동향, 정책 예산 규모, Google Trends 검색 관심도 등)**를 적극적으로 활용하여 예측의 근거를 제시해야 합니다.

**입력 정보:**

1.  **핵심 기술 요약 (이전 분석 결과):**
    {tech_summary_str}

2.  **Google Trends 데이터:**
    {google_trends_str}

3.  **정량적/정성적 데이터 하이라이트:**
    * **연구 동향:**
        {research_quantitative_str}
        {research_qualitative_highlights_str}
    * **뉴스 및 시장 반응 (기술 뉴스 및 기업 IR 포함):**
        {news_quantitative_str}
        {news_qualitative_highlights_str}
    * **정책 및 예산 동향:**
        {policy_quantitative_str}
        {policy_qualitative_highlights_str}

**출력 지침:**
각 핵심 기술에 대해 다음 항목을 포함하여 JSON 형식으로 예측 결과를 제시해주세요:
-   `technology_name`: 기술명 (핵심 기술 요약에서 언급된 기술 대상)
-   `short_term_trend` (향후 1년): "급성장", "성장", "유지", "점진적 감소", "급감" (정량적 근거 명시)
-   `mid_term_trend` (향후 2-3년): "급성장", "성장", "유지", "점진적 감소", "급감" (정량적 근거 명시)
-   `key_driving_factors`: 성장의 주요 동인 (예: "최근 2년간 관련 논문 수 50% 증가", "주요 기업들의 연간 R&D 투자액 X억 달러 발표", "정부의 Y분야 예산 Z% 증액", "Google 검색 관심도 30% 증가")
-   `key_inhibiting_factors`: 성장의 저해 요인 (예: "높은 초기 도입 비용", "관련 전문 인력 부족", "부정적 뉴스 언급 빈도 증가", "Google 검색 관심도 하락")
-   `public_interest_score` (0.0 ~ 10.0): Google Trends 데이터를 기반으로 한 대중 관심도 점수
-   `quantitative_evidence_summary`: 예측에 사용된 주요 정량적 지표 요약 (예: "논문 연평균 20편 발간, 최근 뉴스 100건 중 70% 긍정적 언급, A기업 투자 발표, Google 관심도 7.5/10")
-   `confidence_score` (0.0 ~ 1.0): 예측 신뢰도 (정량적 근거의 명확성에 따라 조정)
-   `prediction_narrative`: 정량적/정성적 근거를 종합한 예측에 대한 상세 설명.

또한, 전체 기술 분야의 종합적인 트렌드 예측 점수 (overall_trend_score, 0-100점)와 그 근거(주요 정량적 지표 포함)도 포함해주세요.

**JSON 출력 예시:**
{{
  "predicted_trends": [
    {{
      "technology_name": "양자 컴퓨팅",
      "short_term_trend": "성장",
      "mid_term_trend": "급성장",
      "key_driving_factors": ["최근 3년간 관련 논문 연평균 30% 증가", "정부의 양자 기술 투자 예산 5년간 1조원 편성", "빅테크 기업들의 프로토타입 공개", "Google 검색 관심도 25% 증가"],
      "key_inhibiting_factors": ["높은 기술적 난이도", "상용화까지의 불확실성"],
      "public_interest_score": 8.5,
      "quantitative_evidence_summary": "논문 증가율 연 30%, 정부 예산 1조원, 주요 기업 투자 활발, Google 검색 관심도 8.5/10",
      "confidence_score": 0.75,
      "prediction_narrative": "단기적으로는 연구 개발 중심의 성장이 예상되나, 정부 및 기업의 대규모 투자와 지속적인 논문 발표 증가로 볼 때 중기적으로 급성장할 잠재력이 매우 높다. 특히 Google Trends에서 확인된 높은 대중적 관심도(8.5/10)는 이 기술의 상업적 잠재력을 보여준다. 다만, 기술적 허들 극복이 관건이다."
    }}
  ],
  "overall_trend_score": 80,
  "overall_trend_reasoning": "다수의 핵심 기술 분야에서 논문 발간 수 증가, 기업 투자 확대, 정부의 정책적 지원 및 예산 투입이 확인되어 전반적으로 강력한 성장 모멘텀이 예측됩니다. 특히 AI 반도체 및 로봇 자동화 분야의 성장세가 두드러집니다. Google Trends 데이터에서도 주요 기술들에 대한 대중 관심도가 전반적으로 증가하는 추세를 보입니다."
}}
"""

class TrendPredictionAgent:
    def __init__(self, model_name=DEFAULT_MODEL):
        self.llm = ChatOpenAI(
            api_key=OPENAI_API_KEY,
            model=model_name,  # Consider gpt-4-turbo for complex reasoning
            temperature=0.2
        )
        self.prompt = PromptTemplate.from_template(TREND_PREDICTION_PROMPT_TEMPLATE)
        self.chain = self.prompt | self.llm | JsonOutputParser()

    def _format_tech_summary_for_prompt(self, tech_summary: Dict) -> str:
        if not tech_summary or "key_technologies" not in tech_summary:
            return "핵심 기술 요약 정보 없음."
        summary_parts = [f"종합 기술 요약: {tech_summary.get('overall_summary', 'N/A')}"]
        for tech in tech_summary.get("key_technologies", []):
            name = tech.get("name", "N/A")
            stage = tech.get("current_stage", "N/A") 
            desc = tech.get("description", "N/A")
            apps = ", ".join(tech.get("applications", ["N/A"]))
            summary_parts.append(f"- {name} (현황: {stage}, 응용분야: {apps}): {desc}")
        return "\n".join(summary_parts)

    def _format_google_trends_data(self, tech_summary: Dict) -> str:
        """Google Trends 데이터를 프롬프트용 문자열로 변환"""
        if not tech_summary or "key_technologies" not in tech_summary:
            return "Google Trends 데이터 없음."
            
        trends_parts = []
        
        # 관련 검색어 정보
        if "related_trend_queries" in tech_summary:
            related_queries = tech_summary.get("related_trend_queries", [])
            if related_queries:
                trends_parts.append(f"주요 관련 검색어: {', '.join(related_queries)}")
        
        # 각 기술별 Google Trends 데이터
        for tech in tech_summary.get("key_technologies", []):
            name = tech.get("name", "N/A")
            google_trends = tech.get("google_trends", {})
            
            if google_trends:
                avg_interest = google_trends.get("avg_interest", "N/A")
                current_interest = google_trends.get("current_interest", "N/A")
                trend_direction = google_trends.get("trend_direction", "N/A")
                
                direction_text = "상승 중" if trend_direction == "up" else "하락 중"
                trends_parts.append(f"- {name}: 평균 관심도 {avg_interest}/100, 현재 관심도 {current_interest}/100 (추세: {direction_text})")
        
        if not trends_parts:
            return "수집된 Google Trends 데이터 없음."
            
        return "\n".join(trends_parts)

    def _extract_quantitative_and_qualitative(self, data: Dict, data_name: str, max_items=3, max_length=150) -> tuple[str, str]:
        """Extracts quantitative metrics and qualitative highlights."""
        quantitative_parts = []
        qualitative_highlights = []

        if not data:
            return f"{data_name} 정량적 데이터 없음.", f"{data_name} 정성적 하이라이트 없음."

        # Example: Research Data
        if "papers" in data and "metrics" in data: # research_data
            metrics = data.get("metrics", {})
            quantitative_parts.append(f"총 논문 수: {metrics.get('total_papers', 'N/A')}")
            if metrics.get("research_topics"):
                 quantitative_parts.append(f"주요 연구 주제: {', '.join(metrics.get('research_topics',[]))}")

            for paper in data.get("papers", [])[:max_items]:
                qualitative_highlights.append(f"- {paper.get('title', 'N/A')}: {(paper.get('abstract', '') or '')[:max_length]}...")
        
        # Example: News Data (tech_news_analysis and company_ir_analysis)
        elif "tech_news_analysis" in data or "company_ir_analysis" in data: # Combined news_data
            tech_news_analysis = data.get("tech_news_analysis", {})
            company_ir_analysis = data.get("company_ir_analysis", {})
            
            total_tech_news_items = sum(len(v.get("news_items", [])) for v in tech_news_analysis.values())
            quantitative_parts.append(f"총 기술 뉴스 항목 수 (다수 키워드 종합): {total_tech_news_items}")
            
            for keyword, kw_data in tech_news_analysis.items():
                for item in kw_data.get("news_items", [])[:1]: # Show 1 per keyword
                    qualitative_highlights.append(f"- 기술뉴스({keyword}): {item.get('title', 'N/A')} - {(item.get('content', '') or '')[:max_length]}...")
            
            total_ir_items = sum(len(v.get("ir_items", [])) for v in company_ir_analysis.values())
            quantitative_parts.append(f"총 기업 IR 항목 수: {total_ir_items}")

            for company, c_data in company_ir_analysis.items():
                for item in c_data.get("ir_items", [])[:1]: # Show 1 per company
                     qualitative_highlights.append(f"- 기업IR({company}): {item.get('title', 'N/A')} - {(item.get('content', '') or '')[:max_length]}...")

        # Example: Policy Data
        elif "policies" in data and "metrics" in data: # policy_data
            metrics = data.get("metrics", {})
            quantitative_parts.append(f"총 정책 수: {metrics.get('total_policies', 'N/A')}")
            focus_analysis = metrics.get("focus_analysis", {})
            budget_summary = (focus_analysis.get("budget_allocation_summary", "예산 정보 없음") or "예산 정보 없음")
            quantitative_parts.append(f"정책 예산 요약: {budget_summary}")

            for policy in data.get("policies", [])[:max_items]:
                qualitative_highlights.append(f"- {policy.get('title', 'N/A')} ({policy.get('region', 'N/A')}): {(policy.get('summary', '') or '')[:max_length]}...")

        quantitative_str = "\n".join(quantitative_parts) if quantitative_parts else f"{data_name} 정량적 데이터 요약 없음."
        qualitative_str = "\n".join(qualitative_highlights) if qualitative_highlights else f"{data_name} 주요 내용 하이라이트 없음."
        
        return quantitative_str, qualitative_str

    def run(self, research_data: Dict, news_data: Dict, policy_data: Dict, tech_summary: Dict) -> Dict:
        logger.info("기술 트렌드 예측 시작 ...")

        tech_summary_str = self._format_tech_summary_for_prompt(tech_summary)
        google_trends_str = self._format_google_trends_data(tech_summary)
        
        research_quantitative_str, research_qualitative_highlights_str = self._extract_quantitative_and_qualitative(research_data, "연구 동향")
        news_quantitative_str, news_qualitative_highlights_str = self._extract_quantitative_and_qualitative(news_data, "뉴스 및 시장 반응")
        policy_quantitative_str, policy_qualitative_highlights_str = self._extract_quantitative_and_qualitative(policy_data, "정책 및 예산 동향")

        try:
            response = self.chain.invoke({
                "tech_summary_str": tech_summary_str,
                "google_trends_str": google_trends_str,
                "research_quantitative_str": research_quantitative_str,
                "research_qualitative_highlights_str": research_qualitative_highlights_str,
                "news_quantitative_str": news_quantitative_str,
                "news_qualitative_highlights_str": news_qualitative_highlights_str,
                "policy_quantitative_str": policy_quantitative_str,
                "policy_qualitative_highlights_str": policy_qualitative_highlights_str,
            })
            
            # 응답에 Google Trends 데이터가 포함된 경우, public_interest_score 정보 확인
            if "predicted_trends" in response:
                for trend in response.get("predicted_trends", []):
                    if "public_interest_score" not in trend:
                        # 기본값 설정
                        trend["public_interest_score"] = 5.0
                        logger.warning(f"기술 '{trend.get('technology_name', 'N/A')}'에 대한 public_interest_score가 없어 기본값 5.0으로 설정합니다.")
            
            logger.info("기술 트렌드 예측 완료.")
            return response
        except Exception as e:
            logger.error(f"기술 트렌드 예측 중 오류 발생: {e}")
            return {
                "predicted_trends": [{"technology_name": "분석 실패", "prediction_narrative": str(e)}],
                "overall_trend_score": 0,
                "overall_trend_reasoning": "오류로 인해 트렌드 예측에 실패했습니다."
            }
