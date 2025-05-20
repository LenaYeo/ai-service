# agents/research_agent.py
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datetime import datetime
from typing import Dict, List
import json
import arxiv  # Import the arxiv package

from langchain_openai import ChatOpenAI

from config import OPENAI_API_KEY, DEFAULT_MODEL
from utils.data_utils import save_intermediate_result

class ResearchAgent:
    """arXiv API를 활용한 연구 논문 수집 및 요약 에이전트"""
    
    def __init__(self, model_name=DEFAULT_MODEL, max_docs=5):
        self.llm = ChatOpenAI(api_key=OPENAI_API_KEY, model=model_name)
        self.max_docs = max_docs
        self.client = arxiv.Client()
    
    def get_papers(self, query: str) -> List[Dict]:
        """arXiv에서 관련 논문을 검색하여 핵심 메타데이터만 반환합니다."""
        # 최신 논문을 우선적으로 검색
        search = arxiv.Search(
            query=query,
            max_results=self.max_docs,
            sort_by=arxiv.SortCriterion.SubmittedDate  # 최신 논문 기준 정렬
        )
        
        # 결과를 간결한 형식으로 변환
        papers = []
        for result in self.client.results(search):
            # 핵심 정보만 추출
            paper = {
                'title': result.title,
                'authors': [author.name for author in result.authors],
                'published': result.published.isoformat(),
                'abstract': result.summary,
                'arxiv_id': result.get_short_id()
            }
            
            papers.append(paper)
            print(f"논문 추가: {paper['title']} ({paper['arxiv_id']})")
        
        print(f"총 {len(papers)}개 논문 정보 추출")
        return papers
    
    def generate_paper_summary(self, paper: Dict) -> str:
        """LLM을 사용하여 논문의 초록을 바탕으로 간결한 요약을 생성합니다."""
        title = paper.get('title', '')
        abstract = paper.get('abstract', '')
        
        if not abstract:
            return "요약 정보가 없습니다."
        
        prompt = f"""
        다음은 연구 논문의 제목과 초록입니다:
        
        제목: {title}
        
        초록: {abstract}
        
        위 논문의 핵심 내용을 3-4 문장으로 요약해주세요. 
        이 연구의 목적, 방법, 주요 결과에 초점을 맞추어 간결하게 요약해 주세요.
        """
        
        try:
            response = self.llm.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            print(f"요약 생성 오류: {str(e)}")
            return "요약 생성 중 오류가 발생했습니다."
    
    def process_papers(self, papers: List[Dict]) -> List[Dict]:
        """논문 정보를 처리하고 요약을 생성합니다."""
        processed_papers = []
        
        for paper in papers:
            # LLM을 사용한 요약 생성
            summary = self.generate_paper_summary(paper)
            
            # 처리된 논문 정보 구성
            processed_paper = {
                'title': paper.get('title', ''),
                'authors': paper.get('authors', []),
                'published': paper.get('published', ''),
                'original_abstract': paper.get('abstract', ''),
                'summary': summary,  # LLM으로 생성한 요약
                'arxiv_id': paper.get('arxiv_id', ''),
                'url': f"https://arxiv.org/abs/{paper.get('arxiv_id', '')}"
            }
            
            processed_papers.append(processed_paper)
        
        return processed_papers
    
    async def run(self, keywords: List[str]) -> Dict:
        """키워드 목록으로 arXiv 검색을 실행하고 논문 정보를 수집합니다."""
        all_papers = []
        
        # 각 키워드로 검색 실행
        for keyword in keywords:
            print(f"\n키워드 '{keyword}' 검색 중...")
            papers = self.get_papers(keyword)
            
            # 검색 키워드 정보 추가
            for paper in papers:
                paper['search_keyword'] = keyword
            
            all_papers.extend(papers)
        
        # 중복 제거 (arXiv ID 기준)
        unique_papers = []
        seen_ids = set()
        
        for paper in all_papers:
            arxiv_id = paper.get('arxiv_id')
            if arxiv_id and arxiv_id not in seen_ids:
                seen_ids.add(arxiv_id)
                unique_papers.append(paper)
        
        print(f"중복 제거 후 {len(unique_papers)}개 논문 남음")
        
        # 논문 처리 및 요약 생성
        processed_papers = self.process_papers(unique_papers)
        
        # 결과 구성
        result = {
            "papers": processed_papers,
            "collection_date": datetime.now().isoformat(),
            "keywords_analyzed": keywords,
            "total_papers": len(unique_papers)
        }
        
        # 중간 결과 저장
        save_intermediate_result(result, "research_agent")
        
        return result

# # 테스트 코드
# if __name__ == "__main__":
#     import asyncio
    
#     async def test_research_agent():
#         agent = ResearchAgent(max_docs=3)
#         print("ResearchAgent 테스트 시작")
        
#         results = await agent.run(["Quantum Computing", "Machine Learning"])
        
#         print(f"\n수집된 논문 수: {results['total_papers']}")
        
#         # 논문 요약 출력
#         if results['papers']:
#             print("\n논문 정보 및 요약:")
#             for paper in results['papers'][:1]:  # 첫 번째 논문만 표시
#                 print(f"제목: {paper['title']}")
#                 print(f"저자: {', '.join(paper['authors'])}")
#                 print(f"발행일: {paper['published']}")
#                 print(f"요약: {paper['summary'][:200]}...")
#                 print(f"URL: {paper['url']}")
    
#     asyncio.run(test_research_agent())