import subprocess
import os

os.chdir(r"H:\基于微博数据的社会风险感知与舆情预测系统")

commands = [
    # 配置 Git 用户信息
    'git config user.email "lr0326@users.noreply.github.com"',
    'git config user.name "lr0326"',
    # 添加所有文件
    "git add .",
    # 查看状态
    "git status --short",
    # 提交
    'git commit -m "Initial commit: 基于微博数据的社会风险感知与舆情预测系统"',
    # 设置分支
    "git branch -M main",
    # 推送
    "git push -u origin main"
]

results = []
for cmd in commands:
    results.append(f"\n>>> {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, encoding='utf-8')
        if result.stdout:
            results.append(f"STDOUT: {result.stdout}")
        if result.stderr:
            results.append(f"STDERR: {result.stderr}")
        results.append(f"Return code: {result.returncode}")
    except Exception as e:
        results.append(f"Error: {e}")

with open("git_result.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(results))

print("\n".join(results))

