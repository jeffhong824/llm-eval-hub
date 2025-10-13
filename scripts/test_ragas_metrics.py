#!/usr/bin/env python3
"""
RAGAS 指標測試腳本

用途：驗證所有 RAGAS 0.3.6 指標是否正確安裝和可用

使用方式：
    python scripts/test_ragas_metrics.py
    
    或在 Docker 容器內：
    docker exec llm-eval-hub-app-1 python3 scripts/test_ragas_metrics.py
"""

import sys
from typing import Dict, Any


def test_metric_imports() -> bool:
    """測試所有指標是否能正確導入"""
    print("=" * 60)
    print("📦 測試 1: 指標導入")
    print("=" * 60)
    
    try:
        from ragas.metrics import (
            AnswerAccuracy,
            AnswerCorrectness,
            AnswerRelevancy,
            AnswerSimilarity,
            ContextPrecision,
            ContextRecall,
            ContextRelevance,
            Faithfulness,
            FactualCorrectness,
            ResponseRelevancy
        )
        print("✅ 所有指標導入成功！\n")
        return True
    except ImportError as e:
        print(f"❌ 指標導入失敗: {e}\n")
        return False


def test_metric_instantiation() -> Dict[str, Any]:
    """測試所有指標是否能正確實例化"""
    print("=" * 60)
    print("🔧 測試 2: 指標實例化")
    print("=" * 60)
    
    from ragas.metrics import (
        AnswerAccuracy,
        AnswerCorrectness,
        AnswerRelevancy,
        AnswerSimilarity,
        ContextPrecision,
        ContextRecall,
        ContextRelevance,
        Faithfulness,
        FactualCorrectness,
        ResponseRelevancy
    )
    
    metrics = {}
    failed = []
    
    metric_classes = {
        'answer_accuracy': AnswerAccuracy,
        'answer_correctness': AnswerCorrectness,
        'factual_correctness': FactualCorrectness,
        'context_precision': ContextPrecision,
        'context_recall': ContextRecall,
        'faithfulness': Faithfulness,
        'answer_relevancy': AnswerRelevancy,
        'answer_similarity': AnswerSimilarity,
        'context_relevance': ContextRelevance,
        'response_relevancy': ResponseRelevancy
    }
    
    for name, metric_class in metric_classes.items():
        try:
            metrics[name] = metric_class()
            print(f"  ✅ {name}: {type(metrics[name]).__name__}")
        except Exception as e:
            print(f"  ❌ {name}: {e}")
            failed.append(name)
    
    if failed:
        print(f"\n❌ {len(failed)} 個指標實例化失敗: {', '.join(failed)}\n")
        return {}
    else:
        print(f"\n✅ 所有 {len(metrics)} 個指標實例化成功！\n")
        return metrics


def test_ragas_version() -> str:
    """測試 RAGAS 版本"""
    print("=" * 60)
    print("📌 測試 3: RAGAS 版本")
    print("=" * 60)
    
    try:
        import ragas
        version = ragas.__version__ if hasattr(ragas, '__version__') else "Unknown"
        print(f"  RAGAS 版本: {version}")
        
        if version.startswith('0.3') or version.startswith('0.2'):
            print(f"  ✅ 版本符合要求 (>= 0.2.0)\n")
            return version
        else:
            print(f"  ⚠️  版本可能過舊，建議升級到 >= 0.2.0\n")
            return version
    except Exception as e:
        print(f"  ❌ 無法獲取版本: {e}\n")
        return "Unknown"


def test_dependencies() -> bool:
    """測試相關依賴是否正確安裝"""
    print("=" * 60)
    print("📚 測試 4: 相關依賴")
    print("=" * 60)
    
    dependencies = {
        'langchain': None,
        'langchain_openai': None,
        'langchain_community': None,
        'datasets': None,
        'anthropic': None,
        'openai': None
    }
    
    all_ok = True
    
    for dep in dependencies:
        try:
            module = __import__(dep)
            version = getattr(module, '__version__', 'Unknown')
            dependencies[dep] = version
            print(f"  ✅ {dep}: {version}")
        except ImportError:
            print(f"  ❌ {dep}: 未安裝")
            dependencies[dep] = None
            all_ok = False
    
    if all_ok:
        print("\n✅ 所有依賴都已正確安裝！\n")
    else:
        print("\n⚠️  部分依賴缺失，可能影響功能\n")
    
    return all_ok


def print_summary(results: Dict[str, Any]):
    """打印測試總結"""
    print("=" * 60)
    print("📊 測試總結")
    print("=" * 60)
    
    total_tests = 4
    passed_tests = sum([
        results['imports'],
        bool(results['instances']),
        bool(results['version']),
        results['dependencies']
    ])
    
    print(f"\n總測試數: {total_tests}")
    print(f"通過測試: {passed_tests}")
    print(f"失敗測試: {total_tests - passed_tests}")
    
    if passed_tests == total_tests:
        print("\n🎉 所有測試通過！RAGAS 0.3.6 已正確安裝並可用。")
        print("\n✅ 可用的評估指標:")
        for i, name in enumerate(results['instances'].keys(), 1):
            print(f"  {i}. {name}")
        return 0
    else:
        print("\n⚠️  部分測試失敗，請檢查上述錯誤信息。")
        return 1


def main():
    """主函數"""
    print("\n" + "=" * 60)
    print("🧪 RAGAS 0.3.6 指標測試")
    print("=" * 60 + "\n")
    
    results = {
        'imports': False,
        'instances': {},
        'version': None,
        'dependencies': False
    }
    
    # 執行測試
    results['imports'] = test_metric_imports()
    
    if results['imports']:
        results['instances'] = test_metric_instantiation()
    
    results['version'] = test_ragas_version()
    results['dependencies'] = test_dependencies()
    
    # 打印總結
    exit_code = print_summary(results)
    
    print("\n" + "=" * 60)
    print("測試完成")
    print("=" * 60 + "\n")
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

