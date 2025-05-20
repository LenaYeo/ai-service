# agents/news_agent.py
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datetime import datetime, timedelta # timedelta는 NewsAPI 사용 시 필요, 현재는 IR만 추가하므로 일단 유지
from typing import Dict, List, Tuple, Optional # Tuple, Optional 추가
import json # json 모듈 추가 (결과 저장 시 유용할 수 있음)
import logging

from langchain_core.output_parsers import StrOutputParser, JsonOutputParser # StrOutputParser 추가 (필요시)
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools import Tool
from dotenv import load_dotenv # load_dotenv 추가 (NEWS_API_KEY 등 환경변수 로드 위함)
import requests # requests 추가 (NewsAPI 사용 시 필요)

# config, utils, prompts 임포트 경로 확인 필요
from config import OPENAI_API_KEY, TAVILY_API_KEY, NEWS_API_KEY, DEFAULT_MODEL # NEWS_API_KEY 추가
from utils.data_utils import save_intermediate_result
from prompts.news_prompts import get_tech_trends_summary_prompt, get_ir_analysis_prompt # get_ir_analysis_prompt 추가

# 환경 변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 주요 기술 기업 목록 (IR 검색 기본값으로 사용)
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

class NewsAgent:
    """기술 뉴스 및 기업 IR 자료를 검색하고 분석하는 에이전트"""
    
    def __init__(self, model_name=DEFAULT_MODEL):
        self.llm = ChatOpenAI(
            api_key=OPENAI_API_KEY, 
            model=model_name,
            temperature=0
        )
        
        # Tavily 검색 도구 설정
        self.tavily_search = TavilySearchResults(max_results=7)
        
        # 뉴스 검색 도구 정의
        self.news_search_tool = Tool(
            name="search_tech_news_tavily", # 명확성을 위해 이름 변경
            func=self._search_tech_news,
            description="Tavily를 사용하여 특정 기술 분야 또는 키워드 관련 뉴스를 검색합니다.",
            return_direct=True # 도구 실행 결과를 직접 반환
        )

        # IR 자료 검색 도구 정의 (복원)
        self.ir_search_tool = Tool(
            name="search_company_ir_tavily", # 명확성을 위해 이름 변경
            func=self._search_company_ir,
            description="Tavily를 사용하여 특정 기업의 IR(Investor Relations) 자료, 재무 결과, 투자 관련 뉴스를 검색합니다.",
            return_direct=True # 도구 실행 결과를 직접 반환
        )
        
        # 기술 트렌드 분석 체인 설정
        self.tech_trends_chain = (
            get_tech_trends_summary_prompt()
            | self.llm
            | JsonOutputParser()
        )

        # IR 자료 분석 체인 설정
        self.ir_analysis_chain = (
            get_ir_analysis_prompt()
            | self.llm
            | JsonOutputParser()
        )
    
    def _search_tech_news(self, query: str) -> List[Dict]:
        """Tavily를 사용하여 기술 뉴스를 검색합니다."""
        logger.info(f"Tavily로 기술 뉴스 검색 중: '{query}'")
        search_query = f"{query} latest technology news research developments innovations" # 검색어 구체화
        
        try:
            results = self.tavily_search.invoke(search_query)
        except Exception as e:
            logger.error(f"Tavily 기술 뉴스 검색 중 오류 발생 ('{query}'): {e}")
            return []
            
        news_items = []
        if isinstance(results, list):
            for item in results:
                news_item = {
                    'title': item.get('title', '제목 없음'),
                    'source': self._extract_domain(item.get('url', '')),
                    'url': item.get('url', 'URL 없음'),
                    'content': item.get('content', '')[:1000],  # 내용 일부 확장 (필요에 따라 조절)
                    'search_keyword': query,
                    'retrieved_at': datetime.now().isoformat()
                }
                news_items.append(news_item)
        else:
            logger.warning(f"Tavily 기술 뉴스 검색 결과가 리스트가 아닙니다 ('{query}'). 결과: {results}")

        logger.info(f"'{query}' 관련 기술 뉴스 {len(news_items)}개 발견")
        return news_items

    def _search_company_ir(self, company_name: str) -> List[Dict]:
        """Tavily를 사용하여 기업 IR 자료를 검색합니다."""
        logger.info(f"Tavily로 기업 IR 자료 검색 중: '{company_name}'")
        search_query = f"{company_name} investor relations earnings report financial results technology investment strategy"
        
        try:
            results = self.tavily_search.invoke(search_query)
        except Exception as e:
            logger.error(f"Tavily IR 자료 검색 중 오류 발생 ('{company_name}'): {e}")
            return []

        ir_items = []
        if isinstance(results, list):
            for item in results:
                # IR 관련 키워드를 포함하는지 좀 더 관대하게 확인 (제목 또는 내용)
                content_lower = item.get('content', '').lower()
                title_lower = item.get('title', '').lower()
                if any(kw in title_lower or kw in content_lower for kw in ['investor', 'earning', 'quarterly', 'financial', 'annual report', 'sec filing', 'shareholder']):
                    ir_item = {
                        'title': item.get('title', '제목 없음'),
                        'company': company_name,
                        'source': self._extract_domain(item.get('url', '')),
                        'url': item.get('url', 'URL 없음'),
                        'content': item.get('content', '')[:1500],  # 내용 일부 확장
                        'search_keyword': company_name, # 검색 키워드는 회사명
                        'retrieved_at': datetime.now().isoformat()
                    }
                    ir_items.append(ir_item)
        else:
            logger.warning(f"Tavily IR 자료 검색 결과가 리스트가 아닙니다 ('{company_name}'). 결과: {results}")
            
        logger.info(f"'{company_name}' 관련 IR 자료 {len(ir_items)}개 발견")
        return ir_items
    
    def _extract_domain(self, url: str) -> str:
        """URL에서 도메인 이름을 추출합니다."""
        if not url or not isinstance(url, str):
            return "알 수 없음"
        try:
            from urllib.parse import urlparse
            domain = urlparse(url).netloc
            return domain.replace('www.', '') if domain else "URL 분석 불가"
        except Exception as e:
            logger.warning(f"URL 도메인 추출 중 오류: {url}, 오류: {e}")
            return "URL 분석 중 오류"
    
    def analyze_tech_news(self, tech_keyword: str, news_items: List[Dict]) -> Dict:
        """기술 분야 뉴스를 분석하여 트렌드를 추출합니다."""
        if not news_items:
            logger.warning(f"'{tech_keyword}' 분야의 분석할 뉴스 데이터가 없습니다.")
            return {
                "key_developments": ["데이터 없음"], "main_players": ["데이터 없음"],
                "emerging_technologies": ["데이터 없음"], "challenges": ["데이터 없음"],
                "future_directions": ["데이터 없음"]
            }
        
        news_data = "\n\n".join([
            f"제목: {item.get('title', '')}\n출처: {item.get('source', '')}\n내용 요약: {item.get('content', '')}" 
            for item in news_items
        ])
        
        try:
            result = self.tech_trends_chain.invoke({
                "tech_field": tech_keyword, # 프롬프트에 맞는 변수명 사용
                "news_data": news_data
            })
            return result
        except Exception as e:
            logger.error(f"'{tech_keyword}' 뉴스 분석 중 오류 발생: {str(e)}")
            return {
                "key_developments": ["분석 실패"], "main_players": ["분석 실패"],
                "emerging_technologies": ["분석 실패"], "challenges": ["분석 실패"],
                "future_directions": ["분석 실패"]
            }

    def analyze_company_ir(self, company_name: str, ir_items: List[Dict]) -> Dict:
        """기업 IR 자료를 분석합니다."""
        if not ir_items:
            logger.warning(f"'{company_name}'의 분석할 IR 자료가 없습니다.")
            return {
                "company": company_name, "tech_investments": ["데이터 없음"],
                "rd_focus": ["데이터 없음"], "collaborations": ["데이터 없음"],
                "future_plans": ["데이터 없음"],
                "financial_metrics": {"r_and_d_spending": "데이터 없음", "revenue_from_new_tech": "데이터 없음", "growth_rate": "데이터 없음"}
            }

        ir_data = "\n\n".join([
            f"문서 제목: {item.get('title', '')}\n출처: {item.get('source', '')}\n내용 요약: {item.get('content', '')}"
            for item in ir_items
        ])

        try:
            result = self.ir_analysis_chain.invoke({
                "company_name": company_name,
                "ir_data": ir_data
            })
            # 결과에 company_name이 누락될 수 있으므로 추가
            if 'company' not in result:
                result['company'] = company_name
            return result
        except Exception as e:
            logger.error(f"'{company_name}' IR 자료 분석 중 오류 발생: {str(e)}")
            return {
                "company": company_name, "tech_investments": ["분석 실패"],
                "rd_focus": ["분석 실패"], "collaborations": ["분석 실패"],
                "future_plans": ["분석 실패"],
                "financial_metrics": {"r_and_d_spending": "분석 실패", "revenue_from_new_tech": "분석 실패", "growth_rate": "분석 실패"}
            }
    
    async def run(self, keywords: List[str], companies: Optional[List[str]] = None) -> Tuple[Dict, Dict]:
        """
        입력된 키워드에 맞는 기술 뉴스를 검색/분석하고, 지정된 기업들의 IR 정보를 검색/분석합니다.
        Args:
            keywords (List[str]): 기술 뉴스 검색 및 분석을 위한 키워드 리스트.
            companies (Optional[List[str]]): IR 정보 검색 및 분석을 위한 기업명 리스트.
                                            None일 경우 기본 TECH_COMPANIES 목록 사용.
        Returns:
            Tuple[Dict, Dict]: 기술 뉴스 결과와 기업 IR 결과 딕셔너리를 포함하는 튜플.
        """
        logger.info(f"기술 뉴스 검색 시작 (키워드: {keywords})")
        if companies is None:
            companies_to_search = TECH_COMPANIES[:5] # 기본값으로 상위 5개 기업
            logger.info(f"IR 검색 대상 기업이 지정되지 않아 기본 목록 사용: {companies_to_search}")
        else:
            companies_to_search = companies
            logger.info(f"기업 IR 검색 시작 (대상: {companies_to_search})")

        # 1. 기술 분야별 뉴스 검색 및 분석
        tech_news_results = {}
        for keyword in keywords:
            # Tavily로 기술 뉴스 검색
            # self.news_search_tool.run()은 내부적으로 _search_tech_news(keyword)를 호출
            # _search_tech_news는 List[Dict]를 반환해야 함
            news_items = self.news_search_tool.run(tool_input=keyword, config={}) # tool_input으로 전달
            
            # 뉴스 분석
            analysis = self.analyze_tech_news(keyword, news_items)
            
            tech_news_results[keyword] = {
                "news_items": news_items,
                "analysis": analysis
            }
            logger.info(f"'{keyword}' 키워드 기술 뉴스 분석 완료: {len(news_items if news_items else [])}개 뉴스 항목")
        
        # 2. 기업 IR 자료 검색 및 분석
        company_ir_results = {}
        if companies_to_search: # 검색할 회사가 있을 경우에만 실행
            for company_name in companies_to_search:
                # self.ir_search_tool.run()은 내부적으로 _search_company_ir(company_name)를 호출
                # _search_company_ir는 List[Dict]를 반환해야 함
                ir_items = self.ir_search_tool.run(tool_input=company_name, config={}) # tool_input으로 전달
                
                # IR 자료 분석
                analysis = self.analyze_company_ir(company_name, ir_items)
                
                company_ir_results[company_name] = {
                    "ir_items": ir_items,
                    "analysis": analysis
                }
                logger.info(f"'{company_name}' 기업 IR 자료 분석 완료: {len(ir_items if ir_items else [])}개 항목")
        else:
            logger.info("IR 검색 대상 기업이 없어 IR 검색 및 분석을 건너뜁니다.")

        # 결과 취합 및 저장
        final_result_data = {
            "tech_news_analysis": tech_news_results, # 키 이름 명확화
            "company_ir_analysis": company_ir_results, # 키 이름 명확화
            "collection_date": datetime.now().isoformat()
        }
        
        save_intermediate_result(final_result_data, "news_agent_combined_analysis") # 파일명 변경
        
        logger.info("모든 뉴스 및 IR 검색/분석 완료")
        
        return tech_news_results, company_ir_results

# # 사용 예시 (test_main.py 등에서 호출 시 참고)
# async def main_test():
#     agent = NewsAgent()
#     tech_keywords = ["Generative AI", "Quantum Computing"]
#     target_companies = ["NVIDIA", "Microsoft"] 
#     # target_companies = None # 기본값 사용 테스트
    
#     tech_results, ir_results = await agent.run(keywords=tech_keywords, companies=target_companies)
    
#     print("\n--- 기술 뉴스 분석 결과 ---")
#     for keyword, data in tech_results.items():
#         print(f"\n키워드: {keyword}")
#         print(f"  뉴스 항목 수: {len(data['news_items'])}")
#         # print(f"  분석: {json.dumps(data['analysis'], indent=2, ensure_ascii=False)}") # 상세 분석 내용 출력
#         if data['news_items']:
#             print(f"  첫 번째 뉴스 제목: {data['news_items'][0]['title']}")

#     print("\n--- 기업 IR 분석 결과 ---")
#     for company, data in ir_results.items():
#         print(f"\n기업: {company}")
#         print(f"  IR 항목 수: {len(data['ir_items'])}")
#         # print(f"  분석: {json.dumps(data['analysis'], indent=2, ensure_ascii=False)}") # 상세 분석 내용 출력
#         if data['ir_items']:
#             print(f"  첫 번째 IR 문서 제목: {data['ir_items'][0]['title']}")

# if __name__ == '__main__':
#     asyncio.run(main_test())
