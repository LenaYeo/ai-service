# utils/data_utils.py
import os
import json
import pandas as pd
from datetime import datetime
from config import DATA_DIR, OUTPUTS_DIR
from weasyprint import HTML, CSS
from weasyprint.text.fonts import FontConfiguration
import markdown
import tempfile

def save_json_data(data, filename, directory=DATA_DIR):
    """데이터를 JSON 파일로 저장합니다"""
    filepath = os.path.join(directory, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return filepath

def load_json_data(filename, directory=DATA_DIR):
    """JSON 파일에서 데이터를 로드합니다"""
    filepath = os.path.join(directory, filename)
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_intermediate_result(data, agent_name):
    """에이전트의 중간 결과를 저장합니다"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{agent_name}_{timestamp}.json"
    return save_json_data(data, filename, directory=OUTPUTS_DIR)

def create_data_frame(data, columns):
    """데이터를 DataFrame으로 변환합니다"""
    return pd.DataFrame(data, columns=columns)

def merge_data_sources(news_data, research_data, policy_data):
    """여러 데이터 소스를 병합합니다"""
    return {
        "news": news_data,
        "research": research_data,
        "policy": policy_data,
        "timestamp": datetime.now().isoformat()
    }

def markdown_to_pdf(markdown_text, output_path, title="보고서"):
    """마크다운 텍스트를 PDF로 변환합니다"""
    # 마크다운을 HTML로 변환
    html_content = markdown.markdown(markdown_text, extensions=['tables', 'fenced_code'])
    
    # HTML 템플릿 생성
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>{title}</title>
        <style>
            body {{ font-family: 'Noto Sans KR', sans-serif; margin: 2cm; }}
            h1, h2, h3, h4 {{ color: #333; }}
            h1 {{ font-size: 24pt; margin-bottom: 20px; }}
            h2 {{ font-size: 18pt; margin-top: 20px; }}
            h3 {{ font-size: 14pt; }}
            p {{ font-size: 11pt; line-height: 1.5; }}
            table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            code {{ background-color: #f5f5f5; padding: 2px 4px; border-radius: 4px; }}
            pre {{ background-color: #f5f5f5; padding: 10px; border-radius: 4px; overflow-x: auto; }}
            .header {{ text-align: center; margin-bottom: 30px; }}
            .footer {{ text-align: center; margin-top: 30px; font-size: 9pt; color: #666; }}
            @page {{ margin: 2cm; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>{title}</h1>
            <p>생성일: {datetime.now().strftime("%Y년 %m월 %d일")}</p>
        </div>
        {html_content}
        <div class="footer">
            <p>© {datetime.now().year} AI 기술 트렌드 분석 서비스</p>
        </div>
    </body>
    </html>
    """
    
    # 폰트 설정
    font_config = FontConfiguration()
    
    # HTML을 PDF로 변환
    try:
        HTML(string=html_template).write_pdf(
            output_path,
            stylesheets=[CSS(string='body { font-family: sans-serif; }', font_config=font_config)],
            font_config=font_config
        )
        return output_path
    except Exception as e:
        print(f"PDF 변환 중 오류 발생: {e}")
        return None

def save_report_as_pdf(report_content, filename=None, directory=OUTPUTS_DIR):
    """보고서 내용을 PDF 파일로 저장합니다"""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"report_{timestamp}.pdf"
    
    output_path = os.path.join(directory, filename)
    
    # 제목 추출 시도 (첫 번째 # 헤더)
    title = "기술 트렌드 분석 보고서"
    for line in report_content.split('\n'):
        if line.startswith('# '):
            title = line.replace('# ', '').strip()
            break
    
    return markdown_to_pdf(report_content, output_path, title)