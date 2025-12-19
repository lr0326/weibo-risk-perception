"""
测试可视化模块
"""
results = []

try:
    from src.visualization.dashboard import Dashboard
    results.append("✓ Dashboard 模块导入成功")

    d = Dashboard()
    results.append("✓ Dashboard 初始化成功")
    results.append(f"  - 端口: {d.port}")
    results.append(f"  - 主机: {d.host}")
except Exception as e:
    results.append(f"✗ Dashboard 失败: {e}")

try:
    from src.visualization.report_generator import ReportGenerator
    results.append("✓ ReportGenerator 模块导入成功")

    r = ReportGenerator()
    results.append("✓ ReportGenerator 初始化成功")
except Exception as e:
    results.append(f"✗ ReportGenerator 失败: {e}")

results.append("\n测试完成!")

with open("test_viz_result.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(results))

print("\n".join(results))

