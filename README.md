# MiShape 0.1 — polygon 车辆设计工作台

本包包含独立 MiShape 软件、两辆 Porsche 素材、参数化生成器、使用文档及验证记录。旧 AeroShape 仅作为 Python 几何依赖提供，启动器运行 MiShape。

## 在线 HTML 版

访问 **[MiShape 在线工作台](https://yl602019618.github.io/MShape/)**。

GitHub Pages 托管完整静态网页，车辆形变、参数生成和导出在浏览器 Web Worker 内通过 Pyodide 0.29.4 执行同一套 Python 几何代码。首次加载约 40 MB 运行时及车型数据，请等待引擎初始化。上传模型不发送到服务器；浏览器 IndexedDB 缓存工程基准，重要设计仍请点击「保存工程」下载 JSON。批量组合在当前页面会话中生成，刷新前请下载。

桌面浏览器推荐。浏览器版未启用 Numba JIT，生成及大批量任务速度取决于设备；本地 Python 版本更适合大模型和持续计算。

推送 `main` 后，GitHub Actions 自动构建并发布。构建方法：

```sh
python3 pages/build.py
python3 -m http.server 8020 --directory _site
```

静态输出不依赖正在运行的本地服务。`node pages/test-runtime.mjs` 验证 WASM 下的模型加载、形变、OBJ 导出、参数生成和批量生成。

## 启动

macOS 双击本目录 `start.command`，或在终端运行：

```sh
bash start.sh
```

需要 Python 3.11+；首次运行可自动建立虚拟环境并安装依赖。已有依赖后可在本机运行。服务地址为 [http://127.0.0.1:8010](http://127.0.0.1:8010)，按终端 `Ctrl+C` 停止。`bash start.sh --check` 只检查环境。

## 内容

- [快速使用说明](mishape/README.md)
- [完整操作与技术说明](mishape/docs/DELIVERY.md)
- [素材数据说明](mishape/assets/README.md)
- [浏览器验证报告](mishape/docs/verification/browser-report.json)；同目录包含最终界面截图和其他验证记录。
- `examples/MiShape-Carrera-12-Variants.zip`：实际 12 变体 OBJ、配方与质量记录。
- `examples/MiShape-Generated-SUV.glb`：实际生成的 SUV 几何示例。
- `mishape/tests/` 与 `tests/test_generation_controls.mjs`：Python 和生成参数联动测试。
- `SHA256SUMS`：包内文件的 SHA-256；`release-info.json`：发布内容说明。

塑形和生成用于概念设计；当前版本没有制造、公差、完整自交或 CFD 认证。内置车尺寸为待校准估计，界面显示近似材质，原 UV/纹理副本独立保存。软件许可见 LICENSE；用户提供的车辆素材不由该软件许可自动授予再分发权。

开发测试需另外安装 pytest、httpx；Node 测试可通过 `PYTHON` 环境变量指定虚拟环境中的 Python。此包不包含旧界面、CFD 权重、原工作区缓存或测试用重型工程副本。
