# Udeco 分析工作约定

## 工作树与环境

- 默认在现有工作树 `D:\MHLcode\.worktrees\udeco-analysis` 工作；开始前用 `git rev-parse --show-toplevel` 核对路径。仅在用户明确要求时创建或切换工作树、分支。
- 项目 Python、Notebook 执行和测试统一使用 `PPG_sensor_env`：优先执行 `conda run -n PPG_sensor_env <command>`，或先激活该环境。运行前按需核对 `sys.executable`；不要假定命令行中的裸 `python` 属于目标环境。Windows 输出乱码时设置 `PYTHONIOENCODING=utf-8`。

## Notebook 与验证

- 保留用户对现有 Notebook 的代码、Markdown 和参数修改。若输出过大且可重新生成，可在编辑或提交前清除执行输出；只清除输出及执行计数，并核对其余内容未变。
- 优先用空内核执行 Notebook 验证；仅在用户明确需要展示或保存结果时，将执行输出写回 Notebook。
- 先运行少量目标测试与真实样本烟测，再运行 `git diff --check`。完整 pytest 若失败，先区分既有问题和本次改动，再决定处理范围；不擅自修改无关 Notebook。

## Git 与文件权限

- 只按约定的精确文件名暂存。保留用户未提交的工作；不强推、不重置用户改动。
- 保持项目目录及子目录现有的 Owner 和 ACL，尤其是 `.codex`、`.worktrees`。
