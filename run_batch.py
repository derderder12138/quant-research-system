"""
批量量化投研系统 —— 主入口 CLI。
用法：
    python run_batch.py                    # 默认使用 config.yaml
    python run_batch.py --config mycfg.yaml  # 指定配置文件
"""

import sys
import os

# 确保项目根目录在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
from batch_runner import BatchRunner

if __name__ == "__main__":
    # 加载 .env 中的 API 密钥
    load_dotenv()

    # 解析配置文件路径
    config_path = "config.yaml"
    if len(sys.argv) > 2 and sys.argv[1] == "--config":
        config_path = sys.argv[2]
    elif len(sys.argv) > 1:
        config_path = sys.argv[1]

    runner = BatchRunner(config_path)
    summary = runner.run_batch()

    # 返回码：全部成功 = 0，有失败 = 1（便于 CI/CD 集成）
    sys.exit(0 if summary["failed"] == 0 else 1)
