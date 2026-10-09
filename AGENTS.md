# Subagent delegation

Delegate only when a subtask can replace substantial reading or analysis by the main agent, or when it can run independently in parallel. The main agent handles small, tightly coupled tasks directly.

When that threshold is met, prefer the project custom agents:

- Use `test_writer` to write focused pytest unit or regression tests from established requirements and implementation behavior.
- Use `edge_case_reviewer` for read-only analysis of boundary conditions, exceptional inputs, numerical stability, and missing tests.
- Use `code_explorer` for read-only mapping of code structure, call chains, entry points, and related implementations.

Give every subagent an explicit, bounded reading scope, naming the relevant files, directories, symbols, or code paths. Subagents must limit reads to that scope and return concise conclusions to the main agent.

# Initial reading scope

During initial exploration, limit reading to the main `.py` and `.ipynb` files that implement the project functions relevant to the task. Read test functions, `outputs/`, `subject_data/`, `tests/`, `.gitignore`, and other auxiliary files or directories only when necessary for the current task. Apply this scope to both the main agent and subagents.

# Python environment

Run this project in the Anaconda environment `PPG_sensor_env`. Use `conda run -n PPG_sensor_env <command>` for project Python commands, scripts, and tests, or activate the environment with `conda activate PPG_sensor_env` before running them. Install project dependencies into this environment.

# Worktree location

Create every new worktree for this project under `D:\MHLcode\.worktrees`, using a dedicated subdirectory such as `D:\MHLcode\.worktrees\<branch-name>`. Before creating a worktree, resolve its target to an absolute path and verify that it is inside this directory, regardless of the current working directory.

# Windows Codex Sandbox permissions

Preserve the Owner and ACL of the project directory and all subdirectories, especially `.codex` and `.worktrees`. Never change directory ownership to `CodexSandboxOffline` or `CodexSandboxOnline`. If a permission adjustment is necessary, limit it to the smallest required scope and keep the Owner as the current Windows user.

# 当前 Udeco 代码职责

- 当前识别算法在 `udeco_uh_roi_edge_detection.py`；根目录的同名 notebook 是分阶段诊断入口。
- `udeco_analysis.py` 提供共用读取、标定、序号修复和 ACC 定位；`AnalysisConfig` 只保存 ACC 参数。改变共用逻辑时同时验证 ROI 和区间统计，避免重新引入旧 H/M/L 流程。
- `udeco_edge_detection.py` 是当前算法的滤波/LOWESS 依赖，并保留独立局部边沿方法；精简文件前检查实际导入关系。
- 展示 notebook 放在 `figures/`。保留用户的参数、文字和样式；模块与默认素材路径相对本工作树解析，默认 SVG 写入 `figures/assets/`。
- `slide` 仅为内部 `cancelled` 的展示别名；物理示意和应用文案应保留这一限制。

# Notebook 验证与 Git

- 修改路径、依赖或分析模块后，从空内核运行 `scripts/verify_notebooks.py`；该脚本在 `PPG_sensor_env` 中运行五本 notebook，并对模块所属工作树作校验。
- 用 `python -m pytest` 运行测试；Windows 输出乱码时设置 `PYTHONIOENCODING=utf-8` 和 `PYTHONUTF8=1`。
- 执行副本和报告保存在 `tmp/`。交付 notebook 清空输出及执行计数，保留代码、Markdown 和配置。
- 原 H/M/L 代码保存在 `codex/udeco-hml-legacy`。更新算法、入口或目录时同步更新 README。
- 保留用户未提交改动及本地 `.codex/`；按本次任务涉及的精确文件名暂存，推送遵循用户授权。
