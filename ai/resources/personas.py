"""Predefined personas and perspectives for document generation."""

from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class Persona:
    """Persona definition."""
    id: str
    name: str
    description: str
    perspective: str
    expertise_areas: List[str]
    communication_style: str
    priority: int  # Lower number = higher priority/generality


# 20 predefined personas ordered by generality/priority
PERSONAS: List[Persona] = [
    Persona(
        id="general_user",
        name="一般用戶",
        description="普通用戶，對技術不太熟悉，需要簡單易懂的解釋",
        perspective="從用戶角度出發，關注實用性和易用性",
        expertise_areas=["基本使用", "常見問題", "用戶體驗"],
        communication_style="簡單直接，避免技術術語",
        priority=1
    ),
    
    Persona(
        id="business_owner",
        name="企業主",
        description="中小企業主，關注成本效益和商業價值",
        perspective="從商業角度評估技術解決方案",
        expertise_areas=["成本分析", "ROI", "商業流程", "競爭優勢"],
        communication_style="專業但易懂，強調商業價值",
        priority=2
    ),
    
    Persona(
        id="technical_manager",
        name="技術主管",
        description="技術團隊負責人，需要了解技術架構和實施細節",
        perspective="從技術管理角度評估解決方案",
        expertise_areas=["技術架構", "團隊管理", "項目實施", "風險評估"],
        communication_style="技術性強，注重細節和可行性",
        priority=3
    ),
    
    Persona(
        id="end_user",
        name="終端用戶",
        description="實際使用產品的用戶，關注功能和體驗",
        perspective="從實際使用體驗出發",
        expertise_areas=["功能需求", "用戶界面", "操作流程", "問題解決"],
        communication_style="實用導向，關注具體操作",
        priority=4
    ),
    
    Persona(
        id="customer_service",
        name="客服代表",
        description="客戶服務人員，需要處理用戶問題和投訴",
        perspective="從客戶服務角度提供支持",
        expertise_areas=["問題診斷", "解決方案", "客戶溝通", "服務流程"],
        communication_style="耐心細緻，善於解釋和引導",
        priority=5
    ),
    
    Persona(
        id="product_manager",
        name="產品經理",
        description="產品負責人，關注產品功能和市場需求",
        perspective="從產品角度分析需求和功能",
        expertise_areas=["需求分析", "功能設計", "市場調研", "用戶反饋"],
        communication_style="結構化思維，注重邏輯和數據",
        priority=6
    ),
    
    Persona(
        id="developer",
        name="開發者",
        description="軟件開發人員，關注技術實現和代碼質量",
        perspective="從開發角度分析技術實現",
        expertise_areas=["編程技術", "系統設計", "代碼質量", "性能優化"],
        communication_style="技術性強，注重實現細節",
        priority=7
    ),
    
    Persona(
        id="data_analyst",
        name="數據分析師",
        description="數據分析專家，關注數據處理和分析",
        perspective="從數據角度分析問題和解決方案",
        expertise_areas=["數據分析", "統計方法", "可視化", "洞察提取"],
        communication_style="數據驅動，注重證據和邏輯",
        priority=8
    ),
    
    Persona(
        id="marketing_specialist",
        name="市場專員",
        description="市場營銷人員，關注品牌推廣和用戶獲取",
        perspective="從營銷角度分析產品和服務",
        expertise_areas=["品牌推廣", "用戶獲取", "內容營銷", "市場策略"],
        communication_style="創意性強，注重用戶心理",
        priority=9
    ),
    
    Persona(
        id="sales_representative",
        name="銷售代表",
        description="銷售人員，需要了解產品優勢和客戶需求",
        perspective="從銷售角度展示產品價值",
        expertise_areas=["產品優勢", "客戶需求", "銷售技巧", "競爭分析"],
        communication_style="說服力強，注重價值展示",
        priority=10
    ),
    
    Persona(
        id="quality_assurance",
        name="質量保證",
        description="QA 測試人員，關注產品質量和缺陷",
        perspective="從質量角度評估產品和服務",
        expertise_areas=["測試方法", "缺陷識別", "質量標準", "改進建議"],
        communication_style="細緻入微，注重細節和標準",
        priority=11
    ),
    
    Persona(
        id="security_expert",
        name="安全專家",
        description="信息安全專家，關注系統安全和隱私保護",
        perspective="從安全角度評估風險和防護",
        expertise_areas=["信息安全", "隱私保護", "風險評估", "合規要求"],
        communication_style="謹慎專業，注重風險控制",
        priority=12
    ),
    
    Persona(
        id="compliance_officer",
        name="合規專員",
        description="合規管理人員，關注法規遵循和風險控制",
        perspective="從合規角度評估業務流程",
        expertise_areas=["法規遵循", "風險管理", "審計要求", "合規流程"],
        communication_style="嚴謹規範，注重合規性",
        priority=13
    ),
    
    Persona(
        id="training_specialist",
        name="培訓專員",
        description="培訓和發展專家，關注知識傳遞和技能提升",
        perspective="從教育角度設計學習內容",
        expertise_areas=["課程設計", "教學方法", "技能評估", "學習路徑"],
        communication_style="教育性強，注重學習效果",
        priority=14
    ),
    
    Persona(
        id="support_engineer",
        name="支持工程師",
        description="技術支持工程師，需要解決複雜技術問題",
        perspective="從技術支持角度提供解決方案",
        expertise_areas=["故障排除", "系統維護", "技術文檔", "用戶培訓"],
        communication_style="技術性強，注重問題解決",
        priority=15
    ),
    
    Persona(
        id="ux_designer",
        name="UX 設計師",
        description="用戶體驗設計師，關注用戶界面和交互設計",
        perspective="從設計角度優化用戶體驗",
        expertise_areas=["界面設計", "交互設計", "用戶研究", "可用性測試"],
        communication_style="創意性強，注重用戶感受",
        priority=16
    ),
    
    Persona(
        id="operations_manager",
        name="運營經理",
        description="運營管理人員，關注流程優化和效率提升",
        perspective="從運營角度分析流程和效率",
        expertise_areas=["流程優化", "效率提升", "資源管理", "績效監控"],
        communication_style="管理導向，注重效率和結果",
        priority=17
    ),
    
    Persona(
        id="financial_analyst",
        name="財務分析師",
        description="財務分析專家，關注成本控制和財務效益",
        perspective="從財務角度評估投資回報",
        expertise_areas=["成本分析", "財務建模", "投資評估", "預算規劃"],
        communication_style="數據驅動，注重財務指標",
        priority=18
    ),
    
    Persona(
        id="legal_counsel",
        name="法律顧問",
        description="法律專業人士，關注法律風險和合規要求",
        perspective="從法律角度評估風險和責任",
        expertise_areas=["法律風險", "合同審查", "合規要求", "爭議解決"],
        communication_style="嚴謹專業，注重法律準確性",
        priority=19
    ),
    
    Persona(
        id="industry_expert",
        name="行業專家",
        description="特定行業的資深專家，具有豐富的行業經驗",
        perspective="從行業角度提供專業見解",
        expertise_areas=["行業趨勢", "最佳實踐", "行業標準", "專業知識"],
        communication_style="專業權威，注重行業洞察",
        priority=20
    )
]


def get_personas(count: int) -> List[Persona]:
    """
    Get personas based on count.
    
    Args:
        count: Number of personas to return
        
    Returns:
        List of personas, cycling through the predefined list if count > 20
    """
    if count <= 0:
        return []
    
    # Sort by priority (lower number = higher priority)
    sorted_personas = sorted(PERSONAS, key=lambda p: p.priority)
    
    # Cycle through personas if count > 20
    result = []
    for i in range(count):
        persona_index = i % len(sorted_personas)
        result.append(sorted_personas[persona_index])
    
    return result


def get_persona_by_id(persona_id: str) -> Persona:
    """Get persona by ID."""
    for persona in PERSONAS:
        if persona.id == persona_id:
            return persona
    raise ValueError(f"Persona with ID '{persona_id}' not found")


def get_persona_names(count: int) -> List[str]:
    """Get persona names for the given count."""
    personas = get_personas(count)
    return [f"{p.name} ({p.description})" for p in personas]


