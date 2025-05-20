class AgentState(dict):
    """
    워크플로우 전체의 상태를 저장하고 관리하는 클래스입니다.
    각 에이전트 간의 정보 공유 및 워크플로우 진행 상황을 나타냅니다.
    """
    initial_query: str = ""                 # 사용자가 입력한 원본 쿼리
    current_keywords: List[str] = []        # 현재 데이터 수집 및 분석에 사용되는 키워드
    research_data_output: Dict[str, Any] = {}       
    tech_news_output: Dict[str, Any] = {}           
    company_ir_output: Dict[str, Any] = {}          
    policy_data_output: Dict[str, Any] = {}         
    tech_summary_analysis: Dict[str, Any] = {}      
    trend_prediction_analysis: Dict[str, Any] = {}  
    risk_opportunity_analysis: Dict[str, Any] = {}  
    data_collection_iteration_count: int = 0      
    max_data_collection_iterations: int = 2       
    validation_attempt_count: int = 0             # 현재 검증 시도 횟수
    max_validation_attempts: int = 2              # 최대 검증 시도 횟수
    last_validation_issues: List[str] = []        
    is_current_analysis_valid: bool = True        
    process_log: List[str] = []                   
    final_report_content: str = ""                
