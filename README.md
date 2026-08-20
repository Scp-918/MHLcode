# Udeco 状态分析

本目录独立实现多受试者 Udeco2/Udeco3 平台状态分离，不修改或依赖参考仓库运行时代码。

## 运行环境

使用 Anaconda 环境 `PPG_sensor_env`。Windows 下应通过 Conda 激活或 `conda run` 启动，确保 SciPy 所需 DLL 路径正确：

```powershell
D:\Code\Minianaconda\Scripts\conda.exe run -n PPG_sensor_env pytest -q
D:\Code\Minianaconda\Scripts\conda.exe run -n PPG_sensor_env jupyter lab udeco_state_analysis.ipynb
```

若新建环境，可安装 `requirements.txt`。Excel 写出使用 `openpyxl`。

## 数据契约

```text
DATA_ROOT/
├── ck.mat
├── ts/
│   ├── ts_write_1_sensor.csv
│   └── ts_write_1_HRdata.csv
├── lyx/
├── lzj/
├── cgx/
├── pjy/
└── hjy/
```

运动名限定为 `write`、`gripper`、`run`、`rope`。批处理只接受严格成对文件；HRdata 内容不参与 Udeco 分类。

## 推荐顺序

1. 在 Notebook 用户参数区设置 `DATA_ROOT` 与 `VALIDATION_FILE`。
2. 保持 `RUN_BATCH=False`，检查单文件读取、H/M/L 主图和 debug 图。
3. 必要时只调整集中参数区并重复验证。
4. 确认后设置 `RUN_BATCH=True`，运行批处理、候选样本检查和 Excel 输出。

当前仓库未包含真实受试者数据，因此自动测试使用确定性合成数据；科研数据结论需在真实文件上完成上述人工验证。
