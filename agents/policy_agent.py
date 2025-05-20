import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datetime import datetime
from typing import Dict, List, Any, Optional
import json
import asyncio
import re
import requests
import logging
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults

# config.py에서 TAVILY_API_KEY를 가져온다고 가정합니다.
# utils.data_utils 및 prompts.policy_prompts도 올바르게 경로 설정되어 있다고 가정합니다.
from config import OPENAI_API_KEY, DEFAULT_MODEL, DATA_DIR, TAVILY_API_KEY
from utils.data_utils import save_intermediate_result
from prompts.policy_prompts import get_policy_analysis_prompt

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PolicyAgent:
    """국가 정책 정보 수집 및 분석 에이전트 (EU API 검색 제외)"""
    
    def __init__(self, model_name=DEFAULT_MODEL, max_policies=20):
        self.llm = ChatOpenAI(api_key=OPENAI_API_KEY, model=model_name)
        self.tavily_search_tool = TavilySearchResults(api_key=TAVILY_API_KEY, max_results=5)
        self.max_policies = max_policies
        
        # 분석 체인 설정
        self.policy_analysis_chain = (
            get_policy_analysis_prompt()
            | self.llm
            | JsonOutputParser()
        )
        
    async def search_policies_from_apis(self, keywords: List[str]) -> List[Dict]:
        """여러 API에서 정책 정보 검색 (미국 API만 사용)"""
        all_policies = []
        
        for keyword in keywords:
            logger.info(f"키워드 '{keyword}'로 정책 검색 중...")
            
            # US Data.gov API 검색
            us_policies = self.search_us_ai_policies(query=keyword, rows=5)
            all_policies.extend(us_policies)
            logger.info(f"미국 정책 {len(us_policies)}개 발견")
            
            # REMOVED: EU 데이터 포털 검색 로직 제거
            
        logger.info(f"API 검색으로 총 {len(all_policies)}개 정책 정보 수집 (미국 API)")
        return all_policies
    
    def search_us_ai_policies(self, query="artificial intelligence policy", rows=5) -> List[Dict]:
        """미국 Data.gov API 검색"""
        try:
            url = "https://catalog.data.gov/api/3/action/package_search"
            params = {"q": query, "rows": rows}
            response = requests.get(url, params=params)
            response.raise_for_status() # HTTP 오류 발생 시 예외 발생
                
            results = response.json().get("result", {}).get("results", [])
            
            return [
                {
                    "title": item.get("title", "No Title"),
                    "region": "USA",
                    "source": "Data.gov",
                    "published_date": item.get("metadata_created", "")[:10], # YYYY-MM-DD 형식으로 가정
                    "url": item.get("url", "N/A"), # Data.gov는 package_show를 통해 접근해야 할 수도 있음
                    "summary": item.get("notes", ""),
                    "policy_type": "정책 데이터", # 또는 item.get("type", "정책 데이터")
                    "search_keyword": query
                    # "impact_level"은 _estimate_impact_level을 사용했었으나, 해당 함수가 없으므로 제거하거나 다시 추가해야 함
                }
                for item in results
                if item.get("notes") # 내용이 있는 항목만 포함
            ]
        except requests.exceptions.RequestException as e:
            logger.error(f"미국 정책 검색 중 네트워크 또는 HTTP 오류 발생: {str(e)}")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"미국 정책 검색 중 JSON 디코딩 오류 발생: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"미국 정책 검색 중 예기치 않은 오류 발생: {str(e)}")
            return []

    # REMOVED: def search_eu_ai_policies(...) 메소드 전체 삭제

    async def search_with_tavily(self, keywords: List[str]) -> List[Dict]:
        """Tavily를 사용하여 정부 정책 및 예산 관련 정보 검색"""
        tavily_query = f"government policies and budget for {', '.join(keywords)}"
        logger.info(f"Tavily 검색 실행: {tavily_query}")
        
        try:
            # TavilySearchResults의 _run 메소드는 동기적이므로 asyncio.to_thread 사용
            raw_results = await asyncio.to_thread(self.tavily_search_tool._run, tavily_query)
            
            processed_policies = []
            for res in raw_results: 
                summary = res.get('content', '')
                if summary: 
                    policy_doc = {
                        "title": res.get('title', 'No Title from Tavily'),
                        "region": "N/A (Web Search)",
                        "source": f"Tavily Search ({res.get('url', 'N/A')})",
                        "published_date": datetime.now().isoformat()[:10], # 웹 검색 결과는 발행일 특정 어려움
                        "url": res.get('url', 'N/A'),
                        "summary": summary,
                        "policy_type": "일반 웹 문서",
                        "search_keyword": ", ".join(keywords),
                        # "impact_level" 관련 로직 필요시 추가
                    }
                    processed_policies.append(policy_doc)
            
            logger.info(f"Tavily가 다음 키워드로 {len(processed_policies)}개 문서를 찾았습니다: {', '.join(keywords)}")
            return processed_policies

        except Exception as e:
            logger.error(f"Tavily 검색 중 오류 발생: {str(e)}")
            return []


    def analyze_policies_by_region(self, policies: List[Dict]) -> Dict[str, int]:
        """지역별 정책 분포 분석"""
        regions = {}
        for policy in policies:
            region = policy.get('region', '기타')
            regions[region] = regions.get(region, 0) + 1
        return dict(sorted(regions.items(), key=lambda x: x[1], reverse=True))
    
    def analyze_policy_timeline(self, policies: List[Dict]) -> Dict[str, int]:
        """정책 발표 시간 추이 분석"""
        timeline = {}
        for policy in policies:
            date_str = policy.get('published_date', '')
            if date_str and isinstance(date_str, str):
                match = re.match(r'(\d{4})-\d{2}-\d{2}', date_str)
                if match:
                    year = match.group(1)
                    timeline[year] = timeline.get(year, 0) + 1
        return dict(sorted(timeline.items(), key=lambda x: x[0]))
    
    async def extract_policy_focus_with_llm(self, policies: List[Dict]) -> Dict[str, Any]: # 반환 타입 Any로 변경
        """LLM을 사용하여 정책의 주요 초점 분야 추출"""
        if not policies:
            return {"focus_areas": [], "key_points": [], "stakeholders": [], "impact": "데이터 없음", "implementation_timeline": "데이터 없음", "international_alignment": "데이터 없음"}
        
        policy_data_entries = []
        for p in policies[:5]: # 상위 5개 정책만 분석
            title = p.get('title', '제목 없음')
            region = p.get('region', '지역 정보 없음')
            summary = p.get('summary', '요약 없음')
            if not isinstance(summary, str): # 요약이 문자열이 아닐 경우 변환
                summary = str(summary)
            policy_data_entries.append(f"제목: {title}\n지역: {region}\n내용: {summary}")
        
        policy_data_str = "\n\n".join(policy_data_entries)

        if not policy_data_str.strip():
             logger.warning("LLM 분석을 위한 정책 데이터가 비어있습니다.")
             return {"focus_areas": ["분석용 데이터 부족"], "key_points": ["분석용 데이터 부족"], "stakeholders": ["분석용 데이터 부족"], "impact": "데이터 없음", "implementation_timeline": "데이터 없음", "international_alignment": "데이터 없음"}

        try:
            # langchain_core.runnables.base.RunnableSerializable 에 ainoke가 있음
            result = await self.policy_analysis_chain.ainvoke({
                "policy_data": policy_data_str
            })
            return result
        except Exception as e:
            logger.error(f"LLM 정책 분석 중 오류 발생: {str(e)}")
            return {
                "focus_areas": ["분석 실패"], "key_points": ["분석 실패"], "stakeholders": ["분석 실패"],
                "impact": "오류로 인한 데이터 없음", "implementation_timeline": "오류로 인한 데이터 없음", "international_alignment": "오류로 인한 데이터 없음"
            }

    async def run(self, keywords: List[str]) -> Dict:
        """키워드 목록으로 정책 정보 검색 및 분석 실행"""
        # 1. API에서 정책 정보 검색
        logger.info(f"키워드로 정책 검색: {keywords}")
        policies = await self.search_policies_from_apis(keywords)
        
        if not policies:
            logger.info("API 검색 결과가 없습니다. Tavily를 사용하여 정부 정책 및 예산 정보를 검색합니다.")
            tavily_policies = await self.search_with_tavily(keywords)
            policies.extend(tavily_policies)
            if tavily_policies:
                logger.info(f"Tavily 검색을 통해 {len(tavily_policies)}개의 추가 문서를 발견했습니다.")
            else:
                logger.info("Tavily 검색으로도 문서를 찾지 못했습니다.")
        
        unique_policies = []
        seen_identifiers = set()
        
        for policy in policies:
            # Ensure title and url are strings before stripping
            # If policy.get('key') returns None (because key exists with None value),
            # (raw_value or '') ensures we operate on an empty string instead of None.
            raw_title = policy.get('title')
            title = (raw_title or '').strip()

            raw_url = policy.get('url')
            url = (raw_url or '').strip()
            
            identifier = (title, url if url else f"no_url_{title}")
            
            if title and identifier not in seen_identifiers: # 제목은 필수
                seen_identifiers.add(identifier)
                unique_policies.append(policy)
        
        logger.info(f"중복 제거 후 {len(unique_policies)}개 정책 문서 남음")
        
        # 정책 분석
        regions = self.analyze_policies_by_region(unique_policies)
        timeline = self.analyze_policy_timeline(unique_policies)
        focus_analysis = await self.extract_policy_focus_with_llm(unique_policies)
        
        # 결과 구성
        result = {
            "policies": unique_policies,
            "metrics": {
                "total_policies": len(unique_policies),
                "policies_by_region": regions,
                "policy_timeline": timeline,
                "focus_analysis": focus_analysis
            },
            "collection_date": datetime.now().isoformat(),
            "keywords_analyzed": keywords,
        }
        
        try:
            # 중간 결과 저장 (DATA_DIR 경로가 올바르게 설정되어 있어야 함)
            # os.makedirs(DATA_DIR, exist_ok=True) # 필요시 DATA_DIR 생성
            serializable_result = json.loads(json.dumps(result, default=str)) # 직렬화 가능한 형태로 변환
            result_path = save_intermediate_result(serializable_result, "policy_agent_run_output")
            logger.info(f"결과 저장 완료: {result_path}")
        except TypeError as te:
            logger.error(f"결과 직렬화 중 오류 발생: {str(te)}. 일부 데이터를 문자열로 변환합니다.")
            # 직렬화 실패 시 대체 저장 로직 (예: 문자열 변환)
            try:
                fallback_result = json.loads(json.dumps(result, default=lambda o: f"<non-serializable: {type(o).__name__}>"))
                result_path = save_intermediate_result(fallback_result, "policy_agent_run_output_fallback")
                logger.info(f"폴백 결과 저장 완료: {result_path}")
            except Exception as e_fallback:
                logger.error(f"폴백 결과 저장 중 오류 발생: {e_fallback}")
        except Exception as e:
            logger.warning(f"결과 저장 중 오류 발생: {str(e)}")
        
        return result

# # 테스트 코드 (필요시 주석 해제하여 사용)
# if __name__ == "__main__":
#     async def main_test():
#         agent = PolicyAgent()
#         test_keywords = ["artificial intelligence national strategy", "AI governance framework"]
#         # test_keywords = ["non_existent_policy_test_keyword_xyz"] # 테스트: 결과 없는 경우
        
#         results = await agent.run(keywords=test_keywords)
        
#         print(f"\n--- 최종 결과 ---")
#         print(f"총 수집된 정책 수: {results.get('metrics', {}).get('total_policies', 0)}")
        
#         if results.get('policies'):
#             print("\n샘플 정책:")
#             for i, policy in enumerate(results['policies'][:2]): # 처음 2개만 출력
#                 print(f"  --- 정책 {i+1} ---")
#                 print(f"  제목: {policy.get('title')}")
#                 print(f"  지역: {policy.get('region')}")
#                 print(f"  출처: {policy.get('source')}")
#                 print(f"  발행일: {policy.get('published_date')}")
#                 # print(f"  요약: {policy.get('summary', '')[:100]}...") # 요약이 길 경우 일부만
#         else:
#             print("수집된 정책이 없습니다.")

#         print("\n지역별 정책 분포:")
#         print(json.dumps(results.get('metrics', {}).get('policies_by_region', {}), indent=2, ensure_ascii=False))
        
#         print("\n연도별 정책 타임라인:")
#         print(json.dumps(results.get('metrics', {}).get('policy_timeline', {}), indent=2, ensure_ascii=False))

#         print("\nLLM 기반 정책 초점 분석:")
#         print(json.dumps(results.get('metrics', {}).get('focus_analysis', {}), indent=2, ensure_ascii=False))

#     asyncio.run(main_test())
