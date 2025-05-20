# agents/report_agent.py
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import Dict, List
import logging
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser # 보고서는 Markdown 문자열로
from langchain_core.prompts import PromptTemplate

from config import OPENAI_API_KEY, DEFAULT_MODEL
# from utils.data_utils import save_intermediate_result # 최종 보고서 저장 로직은 main에 있음

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 프롬프트 정의
REPORT_GENERATION_PROMPT_TEMPLATE = """
당신은 전문 기술 분석 보고서 작성가입니다. 다음 분석 결과들을 바탕으로 종합적인 기술 트렌드 분석 보고서를 Markdown 형식으로 작성해주세요.

보고서 제목: "{report_title} 기술 트렌드 분석 보고서"
작성일: {current_date}

목차:
1.  서론 (분석 배경 및 목적)
2.  주요 기술 동향 요약
    - 핵심 기술 및 현재 상황
    - 종합적인 기술 환경 평가
3.  미래 기술 트렌드 예측
    - 기술별 단기/중기 트렌드 전망
    - 주요 동인 및 잠재적 영향 분야
    - 종합 트렌드 예측 점수 및 근거
4.  주요 리스크 및 기회 요인
    - 식별된 리스크 요인 상세 (영향, 시기, 완화 방안 제안 포함)
    - 식별된 기회 요인 상세 (영향, 시기, 활용 방안 제안 포함)
5.  결론 및 제언

제공된 데이터:
-   **핵심 기술 요약**:
    {tech_summary_str}

-   **예측된 기술 트렌드**:
    {trend_prediction_str}

-   **리스크 및 기회 분석**:
    {risk_opportunity_str}

---
위 정보를 바탕으로, 각 섹션별로 상세하고 통찰력 있는 내용을 포함하여 전문적인 보고서를 작성해주세요.
Markdown 형식을 사용하여 가독성을 높여주세요 (예: 제목은 `##`, `###` 사용, 목록은 `-` 또는 `*` 사용).
"""

class ReportGenerationAgent:
    """최종 분석 보고서를 생성하는 에이전트"""
    def __init__(self, model_name=DEFAULT_MODEL):
        self.llm = ChatOpenAI(
            api_key=OPENAI_API_KEY,
            model=model_name, # 보고서 작성이므로 더 큰 모델 (e.g., gpt-4-turbo) 고려 가능
            temperature=0.4
        )
        self.prompt = PromptTemplate.from_template(REPORT_GENERATION_PROMPT_TEMPLATE)
        self.chain = self.prompt | self.llm | StrOutputParser()

    def _format_input_for_prompt(self, data: Dict, name: str) -> str:
        if not data:
            return f"{name} 정보 없음."
        # 전체 내용을 전달하기 위해 JSON 문자열화 또는 더 상세한 포맷팅
        import json
        # 필요한 부분만 선택적으로 포맷팅하여 전달할 수도 있음
        # 예: tech_summary의 key_technologies 리스트를 보기 좋게 변환
        if name == "핵심 기술 요약" and "key_technologies" in data:
            formatted_str = f"종합 요약: {data.get('overall_summary', 'N/A')}\n핵심 기술:\n"
            for tech in data.get("key_technologies", []):
                formatted_str += f"  - 기술명: {tech.get('name')}\n    단계: {tech.get('current_stage')}\n    응용: {', '.join(tech.get('applications', []))}\n    설명: {tech.get('description')}\n"
            return formatted_str
        
        return json.dumps(data, indent=2, ensure_ascii=False)


    def run(self, query_topic: str, tech_summary: Dict, trend_prediction: Dict, risk_opportunity: Dict) -> str:
        logger.info("최종 보고서 생성 시작...")

        tech_summary_str = self._format_input_for_prompt(tech_summary, "핵심 기술 요약")
        trend_prediction_str = self._format_input_for_prompt(trend_prediction, "예측된 기술 트렌드")
        risk_opportunity_str = self._format_input_for_prompt(risk_opportunity, "리스크 및 기회 분석")
        
        current_date_str = datetime.now().strftime("%Y년 %m월 %d일")

        try:
            report_content = self.chain.invoke({
                "report_title": query_topic, # 초기 쿼리 또는 주요 키워드를 제목으로 활용
                "current_date": current_date_str,
                "tech_summary_str": tech_summary_str,
                "trend_prediction_str": trend_prediction_str,
                "risk_opportunity_str": risk_opportunity_str
            })
            logger.info("최종 보고서 생성 완료.")
            # save_intermediate_result(report_content, "final_report_output.md") # main에서 저장
            return report_content
        except Exception as e:
            logger.error(f"최종 보고서 생성 중 오류 발생: {e}")
            return f"# 보고서 생성 오류\n\n오류 발생: {str(e)}\n\n분석된 데이터를 바탕으로 보고서 생성에 실패했습니다. 수동 검토가 필요합니다."

