# agents/tech_summary_agent.py
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import Dict, List
import logging
import time
from pytrends.request import TrendReq

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

from config import OPENAI_API_KEY, DEFAULT_MODEL

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 프롬프트 정의 (기업 IR 데이터 포함하도록 수정)
SUMMARY_PROMPT_TEMPLATE = """
당신은 AI 기술 분석 전문가입니다. 다음 수집된 연구 논문, 뉴스 기사, 정책 문서, 기업 IR 자료를 바탕으로 현재 논의되는 핵심 AI 기술들을 종합적으로 요약해주세요. 제공된 모든 데이터 출처를 활용하여 균형 잡힌 분석을 제공하되, 특히 기업 IR 자료에서 나타나는 기업들의 투자 방향과 R&D 초점을 반영해주세요.

다음 내용을 포함해야 합니다:
1. 각 핵심 AI 기술의 현재 발전 단계와 성숙도
2. 주요 응용 분야 및 사례
3. 상세 기술 설명 및 주요 특성
4. 관련 주요 기업 및 투자 현황
5. 주요 도전 과제 및 한계점
6. 상업적 잠재력 및 시장 전망
7. 투자 동향 및 흐름

또한 다음 통합적 관점도 제공해주세요:
1. 전체 AI 기술 동향에 대한 종합 요약
2. 기술 간 융합 패턴 및 상호작용
3. 주요 시장 신호 및 움직임
4. 정책 환경이 기술 발전에 미치는 영향
5. 지역별 기술 리더십 및 경쟁 구도

연구 논문 주요 내용:
{research_highlights}

뉴스 기사 주요 내용:
{news_highlights}

정책 문서 주요 내용:
{policy_highlights}

기업 IR(투자자 관계) 자료 주요 내용:
{company_ir_highlights}

주요 기술 분야, 각 기술의 현재 발전 단계, 주요 응용 분야, 관련 기업을 포함하여 JSON 형식으로 답변해주세요.
예시:
{{
  "key_technologies": [
    {{
      "name": "기술명",
      "current_stage": "발전 단계",
      "applications": ["응용 분야 1", "응용 분야 2", "응용 분야 3"],
      "description": "상세 설명",
      "key_companies": ["관련 주요 기업 1", "관련 주요 기업 2"],
      "key_challenges": ["도전 과제 1", "도전 과제 2"],
      "commercial_potential": "상업적 잠재력 평가",
      "investment_trend": "투자 동향",
      "data_sources": ["정보 출처 1", "정보 출처 2"]
    }}
  ],
  "overall_summary": "종합적인 기술 동향 요약",
  "cross_technology_trends": "기술 간 융합 패턴",
  "market_signals": "주요 시장 신호",
  "policy_landscape": "정책 환경의 영향",
  "geographical_leadership": "지역별 기술 리더십"
}}
"""

class TechSummaryAgent:
    """수집된 데이터를 바탕으로 핵심 기술을 요약하는 에이전트"""
    def __init__(self, model_name=DEFAULT_MODEL):
        self.llm = ChatOpenAI(
            api_key=OPENAI_API_KEY,
            model=model_name,
            temperature=0.2
        )
        self.prompt = PromptTemplate.from_template(SUMMARY_PROMPT_TEMPLATE)
        self.chain = self.prompt | self.llm | JsonOutputParser()
        # PyTrends 초기화
        self.pytrends = TrendReq(hl='ko-KR', tz=540)  # 한국어, 한국 시간대

    def _prepare_input_data(self, data: Dict, max_items=3, max_length=300) -> str:
        """데이터를 요약하여 LLM 입력용 문자열로 변환"""
        if not data:
            return "데이터 없음"

        highlights = []
        if "papers" in data:  # research_data
            for paper in data.get("papers", [])[:max_items]:
                title = paper.get('title', '제목 없음')
                abstract = paper.get('abstract', '')[:max_length]
                highlights.append(f"- {title}: {abstract}...")
        elif "news_items" in data:  # news_data (tech_news part)
             for item in data.get("news_items", [])[:max_items]:
                title = item.get('title', '제목 없음')
                content = item.get('content', '')[:max_length]
                highlights.append(f"- {title}: {content}...")
        elif "policies" in data:  # policy_data
            for policy in data.get("policies", [])[:max_items]:
                title = policy.get('title', '제목 없음')
                summary = policy.get('summary', '')[:max_length]
                highlights.append(f"- {title}: {summary}...")
        elif "ir_items" in data:  # company_ir_data
            for item in data.get("ir_items", [])[:max_items]:
                title = item.get('title', '제목 없음')
                company = item.get('company', 'N/A')
                content = item.get('content', '')[:max_length]
                highlights.append(f"- [{company}] {title}: {content}...")
        
        if not highlights:
            return "유의미한 항목 없음"
        return "\n".join(highlights)

    def _prepare_company_ir_highlights(self, company_ir_data: Dict, max_companies=5, max_items_per_company=2) -> str:
        """기업 IR 데이터를 요약하여 문자열로 변환"""
        if not company_ir_data:
            return "기업 IR 데이터 없음"
            
        highlights = []
        company_count = 0
        
        for company, data in company_ir_data.items():
            if company_count >= max_companies:
                break
                
            company_count += 1
            company_items = []
            
            # IR 항목 정보
            if "ir_items" in data and data["ir_items"]:
                for item in data["ir_items"][:max_items_per_company]:
                    title = item.get('title', '제목 없음')
                    content = item.get('content', '')[:300]  # 내용 일부만 표시
                    company_items.append(f"  - 자료: {title}: {content}...")
            
            # IR 분석 정보
            if "analysis" in data:
                analysis = data["analysis"]
                
                # 기술 투자 정보
                tech_investments = analysis.get("tech_investments", [])
                if tech_investments and tech_investments[0] != "데이터 없음" and tech_investments[0] != "분석 실패":
                    company_items.append(f"  - 기술 투자: {', '.join(tech_investments[:3])}")
                
                # R&D 초점
                rd_focus = analysis.get("rd_focus", [])
                if rd_focus and rd_focus[0] != "데이터 없음" and rd_focus[0] != "분석 실패":
                    company_items.append(f"  - R&D 초점: {', '.join(rd_focus[:3])}")
                
                # 재무 지표
                financial_metrics = analysis.get("financial_metrics", {})
                if financial_metrics:
                    rd_spending = financial_metrics.get("r_and_d_spending", "N/A")
                    if rd_spending != "데이터 없음" and rd_spending != "분석 실패":
                        company_items.append(f"  - R&D 지출: {rd_spending}")
            
            if company_items:
                highlights.append(f"기업: {company}")
                highlights.extend(company_items)
                highlights.append("")  # 기업 간 구분을 위한 빈 줄
        
        if not highlights:
            return "유의미한 기업 IR 데이터 없음"
            
        return "\n".join(highlights)

    def _get_google_trends_data(self, tech_names: List[str]) -> Dict:
        """Google Trends API를 사용하여 기술 관심도 데이터 수집"""
        trends_data = {}
        
        try:
            # 5개씩 나누어 처리 (PyTrends API 제한)
            for i in range(0, len(tech_names), 5):
                batch = tech_names[i:i+5]
                if not batch:
                    continue
                    
                # 관심도 데이터 가져오기
                self.pytrends.build_payload(batch, cat=0, timeframe='today 12-m')
                interest_over_time = self.pytrends.interest_over_time()
                
                if not interest_over_time.empty:
                    for tech in batch:
                        if tech in interest_over_time.columns:
                            # 최근 1년간 평균 검색 관심도
                            trends_data[tech] = {
                                'avg_interest': round(interest_over_time[tech].mean(), 2),
                                'current_interest': float(interest_over_time[tech].iloc[-1]),
                                'trend_direction': 'up' if interest_over_time[tech].iloc[-1] > interest_over_time[tech].iloc[-4] else 'down'
                            }
                
                # API 제한 회피를 위한 지연
                time.sleep(1)
                
            # 관련 검색어 (첫 번째 기술에 대해서만)
            if tech_names:
                try:
                    self.pytrends.build_payload([tech_names[0]], cat=0, timeframe='today 12-m')
                    related_queries = self.pytrends.related_queries()
                    
                    if related_queries and tech_names[0] in related_queries:
                        top_queries = related_queries[tech_names[0]].get('top')
                        if top_queries is not None and not top_queries.empty:
                            trends_data['related_queries'] = top_queries.query.tolist()[:5]  # 상위 5개
                except Exception as e:
                    logger.warning(f"관련 검색어 가져오기 실패: {e}")
            
        except Exception as e:
            logger.error(f"Google Trends 데이터 수집 중 오류 발생: {e}")
        
        return trends_data

    def _extract_key_companies_from_ir(self, company_ir_data: Dict) -> Dict[str, List[str]]:
        """기업 IR 자료에서 기술별 주요 기업 추출"""
        tech_companies_map = {}
        
        for company, data in company_ir_data.items():
            if "analysis" not in data:
                continue
                
            analysis = data["analysis"]
            
            # 기술 투자와 R&D 초점 정보를 사용하여 기업이 관심 있는 기술 분야 파악
            tech_investments = analysis.get("tech_investments", [])
            rd_focus = analysis.get("rd_focus", [])
            
            # 기술 투자와 R&D 초점을 합쳐서 고유한 기술 키워드 추출
            tech_keywords = set()
            
            for tech in tech_investments:
                if tech != "데이터 없음" and tech != "분석 실패":
                    tech_keywords.add(tech.lower())
            
            for focus in rd_focus:
                if focus != "데이터 없음" and focus != "분석 실패":
                    tech_keywords.add(focus.lower())
            
            # 각 기술 키워드에 대해 해당 기업 추가
            for tech in tech_keywords:
                if tech not in tech_companies_map:
                    tech_companies_map[tech] = []
                
                if company not in tech_companies_map[tech]:
                    tech_companies_map[tech].append(company)
        
        return tech_companies_map

    def run(self, research_data: Dict, news_data: Dict, policy_data: Dict, company_ir_data: Dict = None) -> Dict:
        """핵심 기술 요약 분석 수행"""
        logger.info("핵심 기술 요약 분석 시작...")

        # news_data에서 tech_news 부분 추출
        tech_news_content = {}
        if news_data and 'tech_news_analysis' in news_data:
            all_tech_news_items = []
            for _keyword, kw_data in news_data['tech_news_analysis'].items():
                if 'news_items' in kw_data:
                    all_tech_news_items.extend(kw_data['news_items'])
            tech_news_content = {"news_items": all_tech_news_items}

        research_highlights = self._prepare_input_data(research_data)
        news_highlights = self._prepare_input_data(tech_news_content)
        policy_highlights = self._prepare_input_data(policy_data)
        
        # 기업 IR 데이터 처리
        company_ir_highlights = "데이터 없음"
        tech_companies_map = {}
        
        if company_ir_data:
            company_ir_highlights = self._prepare_company_ir_highlights(company_ir_data)
            tech_companies_map = self._extract_key_companies_from_ir(company_ir_data)
            logger.info(f"기업 IR 데이터 처리 완료. 기술별 주요 기업 매핑: {len(tech_companies_map)} 항목")

        try:
            # 1단계: 기본 기술 요약 생성
            summary_response = self.chain.invoke({
                "research_highlights": research_highlights,
                "news_highlights": news_highlights,
                "policy_highlights": policy_highlights,
                "company_ir_highlights": company_ir_highlights
            })
            
            # 2단계: 기술별 주요 기업 정보 추가
            if "key_technologies" in summary_response and tech_companies_map:
                for tech in summary_response["key_technologies"]:
                    tech_name = tech.get("name", "").lower()
                    
                    # 유사한 기술명 찾기
                    matched_companies = []
                    for mapped_tech, companies in tech_companies_map.items():
                        # 기술명이 서로의 부분 문자열인 경우 (대소문자 구분 없이)
                        if tech_name in mapped_tech or mapped_tech in tech_name:
                            matched_companies.extend(companies)
                    
                    # 중복 제거
                    matched_companies = list(dict.fromkeys(matched_companies))
                    
                    if matched_companies:
                        # 이미 key_companies가 있으면 병합, 없으면 새로 생성
                        existing_companies = tech.get("key_companies", [])
                        all_companies = existing_companies + [c for c in matched_companies if c not in existing_companies]
                        tech["key_companies"] = all_companies
                    elif "key_companies" not in tech:
                        tech["key_companies"] = []
            
            # 3단계: Google Trends 데이터 추가
            if "key_technologies" in summary_response:
                tech_names = [tech.get("name") for tech in summary_response["key_technologies"] if tech.get("name")]
                trends_data = self._get_google_trends_data(tech_names)
                
                # 기술 요약에 Google Trends 데이터 통합
                for tech in summary_response["key_technologies"]:
                    tech_name = tech.get("name")
                    if tech_name in trends_data:
                        tech["google_trends"] = trends_data[tech_name]
                
                # 전체 요약에 관련 검색어 추가
                if "related_queries" in trends_data:
                    summary_response["related_trend_queries"] = trends_data["related_queries"]
            
            logger.info("핵심 기술 요약 분석 완료 (기업 IR 데이터 및 Google Trends 데이터 포함).")
            return summary_response
        except Exception as e:
            logger.error(f"핵심 기술 요약 분석 중 오류 발생: {e}")
            return {
                "key_technologies": [{"name": "분석 실패", "current_stage": "", "applications": [], "description": str(e), "key_companies": []}],
                "overall_summary": "오류로 인해 요약 생성에 실패했습니다."
            }
