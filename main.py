# main.py
import os
import sys
from datetime import datetime
from typing import Dict, List, Any, Sequence, Optional 
import json
import asyncio 
import logging 

from langgraph.graph import StateGraph, START, END

# 에이전트 임포트
from agents.news_agent import NewsAgent
from agents.research_agent import ResearchAgent
from agents.policy_agent import PolicyAgent 
from agents.tech_summary_agent import TechSummaryAgent
from agents.trend_predict_agent import TrendPredictionAgent 
from agents.risk_opportunity_agent import RiskOpportunityAgent
from agents.validation_agent import ValidationAgent 
from agents.report_agent import ReportGenerationAgent 

# utils 임포트 추가
from utils.data_utils import markdown_to_pdf, save_report_as_pdf

logger = logging.getLogger(__name__)

# 상태 정의 (Refactored AgentState - 필수 정보 중심)
class AgentState(dict):
    """
    워크플로우 전체의 상태를 저장하고 관리하는 클래스입니다.
    각 에이전트 간의 정보 공유 및 워크플로우 진행 상황을 나타냅니다.
    """
    # --- Core Inputs ---
    initial_query: str = ""                 # 사용자가 입력한 원본 쿼리
    current_keywords: List[str] = []        # 현재 데이터 수집 및 분석에 사용되는 키워드
    
    # --- Collected Data Outputs (from data gathering agents) ---
    research_data_output: Dict[str, Any] = {}       
    tech_news_output: Dict[str, Any] = {}           
    company_ir_output: Dict[str, Any] = {}          
    policy_data_output: Dict[str, Any] = {}         
    
    # --- Derived Analysis Products (from analytical agents) ---
    tech_summary_analysis: Dict[str, Any] = {}      
    trend_prediction_analysis: Dict[str, Any] = {}  
    risk_opportunity_analysis: Dict[str, Any] = {}  
    
    # --- Workflow Control & Monitoring ---
    data_collection_iteration_count: int = 0      
    max_data_collection_iterations: int = 2       
    validation_attempt_count: int = 0             # 현재 검증 시도 횟수
    max_validation_attempts: int = 2              # 최대 검증 시도 횟수
    last_validation_issues: List[str] = []        
    is_current_analysis_valid: bool = True        
    
    process_log: List[str] = []                   

    # --- Final Output ---
    final_report_content: str = ""                


# 노드 함수 정의
def query_input_node(state: AgentState) -> Dict[str, Any]:
    query = state.get("initial_query", "") 
    logger.info(f"분석 시작 (원본 쿼리): {query}")
    
    raw_keywords = query.split()
    keywords = [word for word in raw_keywords if len(word.strip()) > 2] 
    
    if not keywords: 
        keywords = [query] if query else []
    
    logger.info(f"추출된 초기 키워드: {keywords}")
    
    return {
        "current_keywords": list(set(keywords)),
        "data_collection_iteration_count": 0, 
        "validation_attempt_count": 0,    
        "process_log": (state.get("process_log",[]) or []) + [f"[{datetime.now()}] 초기 쿼리 처리 완료. 키워드: {keywords}"]
    }

def collect_data_node(state: AgentState) -> Dict[str, Any]:
    """선택적으로 에이전트를 실행하는 데이터 수집 노드"""
    keywords = state["current_keywords"]
    iteration = state["data_collection_iteration_count"]
    is_first_run = iteration == 0
    
    # 간단한 데이터 품질 평가
    research_data = state.get("research_data_output", {})
    news_data = state.get("tech_news_output", {})
    policy_data = state.get("policy_data_output", {})
    
    # 실행 여부 결정 (첫 실행이거나 데이터 부족 시)
    run_research = is_first_run or len(research_data.get("papers", [])) < 5
    run_news = is_first_run or len(news_data) < 3
    run_policy = is_first_run or len(policy_data.get("core_policy_documents", policy_data.get("policies", []))) < 2
    
    # 에이전트 인스턴스화 및 비동기 실행
    async def _run_agents():
        tasks = []
        if run_research: tasks.append(ResearchAgent().run(keywords))
        if run_news: tasks.append(NewsAgent().run(keywords=keywords))
        if run_policy: tasks.append(PolicyAgent().run(keywords))
        return await asyncio.gather(*tasks) if tasks else []
    
    results = asyncio.run(_run_agents())
    if not results:
        logger.info(f"반복 {iteration+1}: 모든 에이전트 충분한 데이터 보유로 실행 건너뜀")
        return {"data_collection_iteration_count": iteration + 1}
    
    # 결과 업데이트 - 이전 상태 복사 후 새 결과만 업데이트
    updated = {
        "data_collection_iteration_count": iteration + 1,
        "research_data_output": research_data,
        "tech_news_output": news_data,
        "company_ir_output": state.get("company_ir_output", {}),
        "policy_data_output": policy_data
    }
    
    # 결과 매핑 (인덱스 관리)
    i = 0
    if run_research: 
        updated["research_data_output"] = results[i]
        logger.info(f"연구 논문 {len(results[i].get('papers', []))}개 수집")
        i += 1
    if run_news:
        tech_results, ir_results = results[i]
        updated["tech_news_output"], updated["company_ir_output"] = tech_results, ir_results
        logger.info(f"뉴스 {len(tech_results)}개, IR {len(ir_results)}개 수집")
        i += 1
    if run_policy:
        updated["policy_data_output"] = results[i]
        key = "core_policy_documents" if "core_policy_documents" in results[i] else "policies"
        logger.info(f"정책 {len(results[i].get(key, []))}개 수집")
    
    log_msg = f"[{datetime.now()}] 반복 {iteration+1} 데이터 수집 완료. 실행: {' '.join(x for x, r in [('연구', run_research), ('뉴스', run_news), ('정책', run_policy)] if r)}"
    updated["process_log"] = (state.get("process_log", []) or []) + [log_msg]
    return updated

def should_rewrite_query_node(state: AgentState) -> str:
    """데이터 수집 반복 여부를 결정"""
    iteration, max_iter = state["data_collection_iteration_count"], state["max_data_collection_iterations"]
    
    # 데이터 충분성 확인
    research_ok = len(state.get("research_data_output", {}).get("papers", [])) >= 5
    news_ok = len(state.get("tech_news_output", {})) >= 3
    policy_key = "core_policy_documents" if "core_policy_documents" in state.get("policy_data_output", {}) else "policies"
    policy_ok = len(state.get("policy_data_output", {}).get(policy_key, [])) >= 2
    data_sufficient = research_ok and news_ok and policy_ok
    
    # 반복 종료 조건: 최대 반복 도달 또는 충분한 데이터 확보
    if iteration >= max_iter or data_sufficient:
        logger.info(f"반복 {iteration}/{max_iter} 종료: {'최대 반복 도달' if iteration >= max_iter else '데이터 충분'}")
        return "analyze_summary"
    else:
        logger.info(f"반복 {iteration}/{max_iter} 계속: 쿼리 재작성으로 진행")
        return "rewrite_query"

def query_rewrite_node(state: AgentState) -> Dict[str, Any]:
    """주제 유지하며 키워드 확장"""
    current = state["current_keywords"]
    
    # 새 키워드 후보 수집
    candidates = set()
    
    # 연구 데이터에서 키워드 추출
    for topic in state.get("research_data_output", {}).get("metrics", {}).get("research_topics", []):
        if topic and isinstance(topic, str) and len(topic.strip()) > 2:
            candidates.add(topic.strip())
    
    # 뉴스 데이터에서 키워드 추출
    for data in state.get("tech_news_output", {}).values():
        # 신흥 기술 키워드 추출
        for tech in data.get("analysis", {}).get("emerging_technologies", []):
            if tech and isinstance(tech, str) and len(tech.strip()) > 2 and tech.lower() not in ["데이터 없음", "분석 실패"]:
                candidates.add(tech.strip())
        
        # 주요 개발사항에서 키워드 추출
        for dev in data.get("analysis", {}).get("key_developments", []):
            if isinstance(dev, str) and dev.lower() not in ["데이터 없음", "분석 실패"]:
                for word in dev.split()[:3]:  # 첫 3단어만 고려
                    if len(word) > 2: 
                        candidates.add(word.strip(',.'))
    
    # 주제 관련성 검사 (간단한 포함 관계 검사)
    filtered = []
    for kw in candidates:
        # 기존 키워드와 중복 방지 및 관련성 검사
        if kw not in current and any(c.lower() in kw.lower() or kw.lower() in c.lower() for c in current):
            filtered.append(kw)
    
    # 최대 5개 키워드 추가 (기존 키워드는 유지)
    added = filtered[:5]
    final = current + added
    final = final[:10]  # 최대 10개로 제한
    
    if not added:
        log = "쿼리 재작성: 새 키워드 없음, 기존 키워드 유지"
    else:
        log = f"쿼리 재작성: 기존 {current} + 추가 {added} → 최종 {final}"
    
    logger.info(log)
    return {
        "current_keywords": final,
        "process_log": (state.get("process_log", []) or []) + [f"[{datetime.now()}] {log}"]
    }

def tech_summary_node(state: AgentState) -> Dict[str, Any]:
    logger.info("핵심 기술 요약 분석 시작...")
    
    tech_summary_agent = TechSummaryAgent()
    combined_news_data_for_summary = {
        "tech_news_analysis": state.get("tech_news_output", {}),
        "company_ir_analysis": state.get("company_ir_output", {})
    }
    tech_summary = tech_summary_agent.run(
        research_data=state.get("research_data_output", {}),
        news_data=combined_news_data_for_summary, 
        policy_data=state.get("policy_data_output", {})
    )
    
    key_tech_count = len(tech_summary.get("key_technologies", []))
    logger.info(f"핵심 기술 {key_tech_count}개 식별 완료.")

    updated_log = (state.get("process_log",[]) or []) + [f"[{datetime.now()}] 핵심 기술 요약 완료. 식별된 기술 수: {key_tech_count}"]
    return {
        "tech_summary_analysis": tech_summary, 
        "process_log": updated_log
    }

def trend_prediction_node(state: AgentState) -> Dict[str, Any]:
    logger.info("트렌드 예측 분석 시작...")
    
    trend_agent = TrendPredictionAgent()
    combined_news_data_for_trend = { 
        "tech_news_analysis": state.get("tech_news_output", {}),
        "company_ir_analysis": state.get("company_ir_output", {})
    }
    trend_prediction = trend_agent.run(
        research_data=state.get("research_data_output", {}),
        news_data=combined_news_data_for_trend, 
        policy_data=state.get("policy_data_output", {}), 
        tech_summary=state.get("tech_summary_analysis", {}) 
    )
    
    overall_score = trend_prediction.get("overall_trend_score", "N/A")
    logger.info(f"트렌드 예측 완료 (전체 점수: {overall_score}).")
    
    updated_log = (state.get("process_log",[]) or []) + [f"[{datetime.now()}] 트렌드 예측 완료. 전체 점수: {overall_score}"]
    return {
        "trend_prediction_analysis": trend_prediction, 
        "process_log": updated_log
    }

def risk_opportunity_node(state: AgentState) -> Dict[str, Any]:
    logger.info("리스크 및 기회 분석 시작...")
        
    risk_opp_agent = RiskOpportunityAgent()
    combined_news_data_for_risk = {
        "tech_news_analysis": state.get("tech_news_output", {}),
        "company_ir_analysis": state.get("company_ir_output", {})
    }
    risk_opportunity = risk_opp_agent.run(
        research_data=state.get("research_data_output", {}),
        news_data=combined_news_data_for_risk, 
        policy_data=state.get("policy_data_output", {}),
        tech_summary=state.get("tech_summary_analysis", {}),    
        trend_prediction=state.get("trend_prediction_analysis", {}), 
        validation_errors=state.get("last_validation_issues", []) 
    )
    
    identified_factors_list = risk_opportunity.get("identified_factors", [])
    if not isinstance(identified_factors_list, list):
        identified_factors_list = []
        logger.warning("RiskOpportunityAgent 결과에서 'identified_factors'가 리스트가 아닙니다. 빈 리스트로 처리합니다.")

    factors_count = len(identified_factors_list) 
    logger.info(f"리스크/기회 요인 {factors_count}개 식별 완료.")
    
    updated_log = (state.get("process_log",[]) or []) + [f"[{datetime.now()}] 리스크/기회 분석 완료. 식별된 요인 수: {factors_count}"]
    return {
        "risk_opportunity_analysis": risk_opportunity, 
        "process_log": updated_log
    }

def validation_node(state: AgentState) -> Dict[str, Any]:
    """분석 결과 검증"""
    current_validation_attempt = state.get('validation_attempt_count', 0) + 1
    max_validation_attempts = state.get('max_validation_attempts', 2)
    
    logger.info(f"분석 결과 검증 시작 (시도: {current_validation_attempt}/{max_validation_attempts})...")
    
    validation_agent = ValidationAgent()
    combined_news_data_for_validation = {
        "tech_news_analysis": state.get("tech_news_output", {}),
        "company_ir_analysis": state.get("company_ir_output", {})
    }
    validation_result_data = validation_agent.run( 
        research_data=state.get("research_data_output", {}),
        news_data=combined_news_data_for_validation, 
        policy_data=state.get("policy_data_output", {}),
        tech_summary=state.get("tech_summary_analysis", {}),
        trend_prediction=state.get("trend_prediction_analysis", {}),
        risk_opportunity=state.get("risk_opportunity_analysis", {})
    )
    
    is_valid = validation_result_data.get("is_valid", False)
    issues = validation_result_data.get("issues", [])
    logger.info(f"검증 결과: {'유효함' if is_valid else '유효하지 않음'}. 이슈: {len(issues)}개")
    if not is_valid:
        logger.warning(f"검증 실패 이슈: {issues}")

    updated_log = (state.get("process_log",[]) or []) + [f"[{datetime.now()}] 검증 시도 {current_validation_attempt} 완료. 결과: {'유효' if is_valid else '유효하지 않음'}. 이슈: {issues if not is_valid else '없음'}"]
    
    return {
        "is_current_analysis_valid": is_valid,       
        "last_validation_issues": issues if not is_valid else [], 
        "validation_attempt_count": current_validation_attempt, 
        "process_log": updated_log
    }

def should_proceed_to_report_node(state: AgentState) -> str:
    """검증 결과에 따라 분기 결정"""
    is_valid = state.get("is_current_analysis_valid", False) 
    current_attempts = state.get("validation_attempt_count", 0) 
    max_attempts = state.get("max_validation_attempts", 2) 
    
    decision = ""
    log_message_for_next_node = "" 

    if is_valid:
        decision = "generate_report"
        log_message_for_next_node = f"[{datetime.now()}] 검증 통과 (시도 {current_attempts}). 보고서 생성으로 진행."
        logger.info(log_message_for_next_node)
    elif current_attempts >= max_attempts:
        decision = "generate_report" 
        log_message_for_next_node = f"[{datetime.now()}] 최대 검증 재시도 횟수({max_attempts}) 도달 (현재 시도 {current_attempts}). 검증에 실패했지만 보고서 생성을 진행합니다."
        logger.warning(log_message_for_next_node)
    else:
        decision = "analyze_risk_opportunity_again"
        log_message_for_next_node = f"[{datetime.now()}] 검증 실패 (시도 {current_attempts}/{max_attempts}). 리스크/기회 분석 재수행을 위해 분기합니다."
        logger.warning(log_message_for_next_node)
    
    return decision 

def report_generation_node(state: AgentState) -> Dict[str, Any]:
    """최종 보고서 생성"""
    logger.info("최종 보고서 생성 시작...")
    
    report_agent = ReportGenerationAgent()
    final_report = report_agent.run(
        query_topic=state.get("initial_query", "기술 트렌드"), 
        tech_summary=state.get("tech_summary_analysis", {}),
        trend_prediction=state.get("trend_prediction_analysis", {}),
        risk_opportunity=state.get("risk_opportunity_analysis", {})
    )
    
    logger.info("최종 보고서 생성 완료!")
    
    # 마크다운 파일로 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    query_topic = state.get("initial_query", "tech_trend").split()[0].lower()
    query_topic = ''.join(c for c in query_topic if c.isalnum())
    md_filename = f"trend_report_{query_topic}_{timestamp}_report.md"
    
    with open(md_filename, "w", encoding="utf-8") as f:
        f.write(final_report)
    logger.info(f"마크다운 보고서가 저장되었습니다: {md_filename}")
    
    # PDF 파일로 변환 (data_utils의 함수 사용)
    pdf_filename = f"trend_report_{query_topic}_{timestamp}.pdf"
    pdf_path = save_report_as_pdf(final_report, pdf_filename)
    logger.info(f"PDF 보고서가 생성되었습니다: {pdf_path}")
    
    updated_log = (state.get("process_log",[]) or []) + [
        f"[{datetime.now()}] 최종 보고서 생성 완료. 마크다운: {md_filename}, PDF: {pdf_path}"
    ]
    
    return {
        "final_report_content": final_report,
        "final_report_md_path": md_filename,
        "final_report_pdf_path": pdf_path,
        "process_log": updated_log
    }

# 워크플로우 그래프 정의
def create_workflow():
    """워크플로우 그래프 생성"""
    workflow = StateGraph(AgentState)
    
    workflow.add_node("query_input", query_input_node)
    workflow.add_node("collect_data", collect_data_node)
    workflow.add_node("query_rewrite", query_rewrite_node)
    workflow.add_node("analyze_tech_summary", tech_summary_node)
    workflow.add_node("predict_trends", trend_prediction_node)
    workflow.add_node("analyze_risk_opportunity", risk_opportunity_node)
    workflow.add_node("validate_analysis", validation_node)
    workflow.add_node("generate_report", report_generation_node)
    
    workflow.add_edge(START, "query_input")
    workflow.add_edge("query_input", "collect_data")
    
    workflow.add_conditional_edges(
        "collect_data",
        should_rewrite_query_node,
        {
            "rewrite_query": "query_rewrite",      
            "analyze_summary": "analyze_tech_summary" 
        }
    )
    workflow.add_edge("query_rewrite", "collect_data") 
    
    workflow.add_edge("analyze_tech_summary", "predict_trends")
    workflow.add_edge("predict_trends", "analyze_risk_opportunity")
    workflow.add_edge("analyze_risk_opportunity", "validate_analysis")
    
    workflow.add_conditional_edges(
        "validate_analysis",
        should_proceed_to_report_node,
        {
            "generate_report": "generate_report",
            "analyze_risk_opportunity_again": "analyze_risk_opportunity" 
        }
    )
    
    workflow.add_edge("generate_report", END)

    return workflow.compile()

def generate_pdf_report(markdown_content: str, output_path: str = None) -> str:
    """마크다운 형식의 보고서를 PDF로 변환하여 저장합니다.
    
    Args:
        markdown_content: 마크다운 형식의 보고서 내용
        output_path: PDF 파일 저장 경로 (None인 경우 자동 생성)
        
    Returns:
        생성된 PDF 파일 경로
    """
    # 출력 경로가 없으면 자동 생성
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # outputs 디렉토리가 없으면 생성
        os.makedirs("outputs", exist_ok=True)
        output_path = f"outputs/trend_report_{timestamp}.pdf"
    
    # 제목 추출 시도 (첫 번째 # 헤더)
    title = "기술 트렌드 분석 보고서"
    for line in markdown_content.split('\n'):
        if line.startswith('# '):
            title = line.replace('# ', '').strip()
            break
    
    # data_utils의 markdown_to_pdf 함수 사용
    return markdown_to_pdf(markdown_content, output_path, title)

def report_generation_node(state: AgentState) -> Dict[str, Any]:
    """최종 보고서 생성"""
    logger.info("최종 보고서 생성 시작...")
    
    report_agent = ReportGenerationAgent()
    final_report = report_agent.run(
        query_topic=state.get("initial_query", "기술 트렌드"), 
        tech_summary=state.get("tech_summary_analysis", {}),
        trend_prediction=state.get("trend_prediction_analysis", {}),
        risk_opportunity=state.get("risk_opportunity_analysis", {})
    )
    
    logger.info("최종 보고서 생성 완료!")
    
    # 마크다운 파일로 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    query_topic = state.get("initial_query", "tech_trend").split()[0].lower()
    query_topic = ''.join(c for c in query_topic if c.isalnum())
    md_filename = f"trend_report_{query_topic}_{timestamp}_report.md"
    
    with open(md_filename, "w", encoding="utf-8") as f:
        f.write(final_report)
    logger.info(f"마크다운 보고서가 저장되었습니다: {md_filename}")
    
    # PDF 파일로 변환 (data_utils의 함수 사용)
    pdf_filename = f"trend_report_{query_topic}_{timestamp}.pdf"
    pdf_path = save_report_as_pdf(final_report, pdf_filename)
    logger.info(f"PDF 보고서가 생성되었습니다: {pdf_path}")
    
    updated_log = (state.get("process_log",[]) or []) + [
        f"[{datetime.now()}] 최종 보고서 생성 완료. 마크다운: {md_filename}, PDF: {pdf_path}"
    ]
    
    return {
        "final_report_content": final_report,
        "final_report_md_path": md_filename,
        "final_report_pdf_path": pdf_path,
        "process_log": updated_log
    }

def main(query: str, max_data_iterations: int = 2, max_validation_attempts: int = 2):
    """분석 워크플로우 실행"""
    if not query:
        logger.error("입력 쿼리가 비어있습니다.")
        return "오류: 분석할 쿼리를 입력해주세요."

    workflow = create_workflow()
    
    initial_state_dict = {
        "initial_query": query,
        "max_data_collection_iterations": max_data_iterations,
        "max_validation_attempts": max_validation_attempts,
        # AgentState 클래스 정의의 기본값들이 나머지 필드에 사용됨
    }
    
    logger.info("워크플로우 실행 시작...")
    # LangGraph의 invoke 메소드에 config 딕셔너리를 통해 recursion_limit 설정
    final_state_dict = workflow.invoke(initial_state_dict, {"recursion_limit": 100}) 

    logger.info("\n--- 최종 워크플로우 상태 요약 ---")
    logger.info(f"원본 쿼리: {final_state_dict.get('initial_query')}")
    logger.info(f"최종 사용 키워드: {final_state_dict.get('current_keywords')}")
    logger.info(f"데이터 수집 반복: {final_state_dict.get('data_collection_iteration_count')}")
    logger.info(f"검증 시도: {final_state_dict.get('validation_attempt_count')}")
    logger.info(f"최종 검증 유효성: {final_state_dict.get('is_current_analysis_valid')}")
    if final_state_dict.get('last_validation_issues'):
        logger.warning(f"최종 검증 이슈: {final_state_dict.get('last_validation_issues')}")
    
    final_report_content = final_state_dict.get("final_report_content")
    if final_report_content:
        logger.info(f"최종 보고서가 성공적으로 생성되었습니다.")
        pdf_path = final_state_dict.get("final_report_pdf_path")
        if pdf_path and os.path.exists(pdf_path):
            logger.info(f"PDF 보고서 경로: {pdf_path}")
            return final_report_content, pdf_path
        else:
            # PDF가 없는 경우 생성
            pdf_path = save_report_as_pdf(final_report_content)
            logger.info(f"PDF 보고서가 생성되었습니다: {pdf_path}")
            return final_report_content, pdf_path
    else:
        logger.error("워크플로우가 최종 보고서를 생성하지 못했습니다.")
        error_summary = {
            "message": "최종 보고서 생성 실패. 중간 결과를 확인하세요.",
            "initial_query": final_state_dict.get("initial_query"),
            "current_keywords": final_state_dict.get("current_keywords"),
            "final_validation_status": final_state_dict.get("is_current_analysis_valid"),
            "final_validation_issues": final_state_dict.get("last_validation_issues"),
            "process_log_tail": (final_state_dict.get("process_log",[]) or [])[-5:] 
        }
        return json.dumps(error_summary, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="AI 기술 트렌드 분석 및 보고서 생성기")
    parser.add_argument("query", type=str, help="분석할 기술 또는 주제 (예: '자율주행 자동차 기술 동향')")
    parser.add_argument("--iterations", dest="max_data_iterations", type=int, default=2, help="데이터 수집 및 쿼리 재작성 최대 반복 횟수")
    parser.add_argument("--validation_retries", dest="max_validation_attempts", type=int, default=2, help="분석 검증 최대 재시도 횟수 (1이면 초기 검증만, 2이면 1회 재시도 후 실패해도 보고서 생성)")
    
    args = parser.parse_args()
    
    if not logging.getLogger().handlers: 
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    logger.info(f"입력 쿼리: {args.query}")
    logger.info(f"데이터 수집 최대 반복 횟수: {args.max_data_iterations}")
    logger.info(f"검증 최대 재시도 횟수 (max_validation_attempts): {args.max_validation_attempts}")

    report_output = main(
        args.query, 
        max_data_iterations=args.max_data_iterations,
        max_validation_attempts=args.max_validation_attempts
    )
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    query_slug = "".join(filter(str.isalnum, args.query[:30])).lower()
    filename_prefix = f"trend_report_{query_slug}_{timestamp}"

    try:
        parsed_json = json.loads(report_output)
        if isinstance(parsed_json, dict) and "message" in parsed_json and "최종 보고서 생성 실패" in parsed_json["message"]:
            filename = f"{filename_prefix}_error.json"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(report_output) 
            logger.info(f"오류/중간 결과 저장 완료: {filename}")
        else: 
            if report_output.strip().startswith("{") and report_output.strip().endswith("}"): 
                 filename = f"{filename_prefix}_report.json"
            else: 
                 filename = f"{filename_prefix}_report.md"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(report_output) 
            logger.info(f"보고서 저장 완료: {filename}")

    except json.JSONDecodeError: 
        filename = f"{filename_prefix}_report.md"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(report_output)
        logger.info(f"보고서 저장 완료: {filename}")
    except Exception as e:
        logger.error(f"보고서 저장 중 예외 발생: {e}")
        fallback_filename = f"{filename_prefix}_unknown_output.txt"
        with open(fallback_filename, "w", encoding="utf-8") as f:
            f.write(str(report_output))
        logger.info(f"알 수 없는 형식의 출력 저장 완료: {fallback_filename}")
