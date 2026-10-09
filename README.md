# Udeco：Uh–ROI 引导的边沿与状态识别

当前 `main` 使用 Uh 定位事件 ROI，再在配对 Udeco 中定位边沿并识别事件。Udeco2、Udeco3 独立处理；固定时间区间统计作为单独流程保留。

## 算法与状态语义

1. 修复 Pydisplay 序号、插值异常值，使用 `ck.mat` 计算两路 Udeco，统一时间轴。
2. Uh/Udeco 使用 4 点中值滤波和 LOWESS 基线估计，默认基线窗口 10 秒、稳健迭代 1 次；前 15 秒提供固定噪声基线。
3. ACC 定位运动段，在运动段内估计 Uh 主频，检峰并构建各自 ROI。运动段和主频可手动覆盖。
4. 在配对 Udeco 的 ROI 及左拓范围内寻找差分极值，定位、补全上升/下降沿，并按幅度与形态检查进行状态识别。

- `detach-attach`：合格边沿配对满足正向尖锐形态及先后顺序条件。
- `release-force`：其他通过配对检查的事件。
- 内部 `cancelled` 表示未通过事件配对分类的 ROI；展示中使用 `slide`。**该别名不代表已独立验证的滑移识别。**

算法输出是 ROI 与边沿事件，不是旧版 H/M/L 平台标签，也不是已经验证的完整腕表模式分类器。SVG 中的物理过程和信号均为示意。

## 文件与职责

```text
udeco_uh_roi_edge_detection.py       # 当前 ROI、边沿定位及事件识别
udeco_uh_roi_edge_detection.ipynb    # 分阶段分析与诊断入口
udeco_edge_detection.py             # 共享滤波/LOWESS，以及独立局部边沿方法
udeco_analysis.py                   # 共用读取、标定、序号修复及 ACC 检测
udeco_interval_analysis.py          # 固定区间统计和 Excel 输出
udeco_interval_statistics.ipynb     # 固定区间统计入口
figures/
  udeco_publication_figures.ipynb    # 图 1–10，共 13 张图
  udeco_flowchart_assets.ipynb       # 生成流程及 LOWESS SVG
  udeco_watch_skin_physics.ipynb     # 手表—皮肤物理过程 SVG
  assets/                           # 本地生成的 SVG，Git 忽略
scripts/verify_notebooks.py          # 五本 notebook 的空内核运行验证
```

## 环境与数据

统一使用 Anaconda 环境 `PPG_sensor_env`。如需安装依赖或注册 Jupyter 内核：

```powershell
conda run -n PPG_sensor_env python -m pip install -r requirements.txt
conda run -n PPG_sensor_env python -m ipykernel install --user --name PPG_sensor_env --display-name PPG_sensor_env
```

数据目录包含 `ck.mat` 和各受试者子目录。单样本分析读取 `*_sensor.csv`；区间统计批处理要求对应的 `*_HRdata.csv`，其内容不参与 Udeco 分类。

默认数据目录为当前工作树的 `subject_data/`；可设置 `UDECO_DATA_ROOT` 指向共享数据目录。默认演示样本为 `cgx/cgx_rope_1_sensor.csv`，采样率为 100 Hz。使用其他数据时同时确认标定和真实采样率。

## 运行顺序

从仓库根目录或 `figures/` 启动 Jupyter，选择 `PPG_sensor_env` 内核：

```powershell
conda run -n PPG_sensor_env jupyter lab
```

- **分析**：打开根目录的 Uh–ROI notebook，修改首个代码单元中的 `SAMPLE_NAME`、`DATA_ROOT` 和参数，再从头运行。阶段参数变化后重跑该阶段及后续单元。
- **展示**：先运行 `figures/udeco_flowchart_assets.ipynb` 生成图 9、10 的素材；再运行 `figures/udeco_publication_figures.ipynb`。在配置单元中填写 `SAMPLE_CSV`，可设置显示时间、运动段及主频覆盖。
- **物理示意**：独立运行 `figures/udeco_watch_skin_physics.ipynb`。
- **区间统计**：运行根目录的区间统计 notebook，先保持 `RUN_BATCH=False` 检查单样本，确认后再启用批处理和 Excel 输出。

展示图保留交互缩放和联动时间轴；图 3 显示 0–10 Hz，图 6 高度 1240 px，图 8 仅含 PPG/Uh/Udeco 三个基线子图。图 9、10 的函数和调用单元可独立于样本分析运行。

SVG 默认写入 `figures/assets/`。修改素材 notebook 的“只改这里”配置，重新生成后再运行展示单元。若覆盖输出目录，同步修改展示 notebook 的 SVG 路径。直接编辑 SVG 的内容会被再次生成覆盖。

## 测试与交付

在仓库根目录执行：

```powershell
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'
conda run --no-capture-output -n PPG_sensor_env python -m pytest -q -p no:cacheprovider
conda run --no-capture-output -n PPG_sensor_env python scripts/verify_notebooks.py
```

验证脚本依次运行分析、区间统计、流程素材、物理示意和专用绘图五本 notebook，每本使用新内核并确认 Python 环境及模块所属工作树。默认测试真实演示样本，在执行副本中关闭区间统计全量批处理，异常直接使验证失败。

执行副本、结果摘要和临时内核配置保存在 `tmp/notebook-verification/`，不修改原 notebook 的配置和输出。可用 `--data-root`、`--sample` 指定样本，或用 `--reference-dir` 对照另一轮的 ROI、边沿和区间统计结果。提交的 notebook 均清空执行输出。

## 旧版分支

旧版 H/M/L 自适应平台分类、状态批处理及相关导出保存在 `codex/udeco-hml-legacy`。需要运行旧流程时，在工作区干净的情况下切换：

```powershell
git switch codex/udeco-hml-legacy
# 返回当前版本
git switch main
```

新版本发布分支为 `codex/udeco-roi-release`。旧 H/M/L API 已从当前共用数据模块移除；依赖旧标签或旧状态 Excel 格式的代码应使用归档分支。
