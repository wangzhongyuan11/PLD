# B 组 FPGA 教材（独立教学产物）

本目录只维护《B组FPGA加速器从零到独立实现教材》；**不参与 A 组模型训练、选模或最终留出集评价**，也不表示 B 组 RTL/上板工作已完成。日常恢复 A 组工作只读 `AGENTS.md` 和精简后的 `docs/A02_当前断点.md`，无需加载本目录正文。

- 成品：`output/pdf/B组FPGA加速器从零到独立实现教材.pdf`（A4，33 页，18 章、3 个附录；2026-09-25 生成；806,082 字节；SHA256 `10729d947dc17960659230fe322c9556292ea661d982899d45141f285be9027c`）。
- 内容源：`docs/b_textbook/B组教材_正文.md`。建构器：`docs/b_textbook/build_pdf.py`。算例只读复核：`docs/b_textbook/verify_examples.py`。结构/缩略图 QA：`docs/b_textbook/qa_pdf.py`。
- 学习范围：从 PS/PL、数字电路、补码和固定点，到 15 层 INT8 数值合同、缓存/分块/PE、RTL、AXI/DMA、golden 验证、综合时序、上板与 B/C 联调。提供八个局部实训和分关验收；没有伪造完整 B 组工程或板级实测。
- PDF 中的项目基线接口取自冻结的 `deliverables/role_a/`；后训练候选若换版，应重新核对规格与 golden。官方来源及文档编号在 PDF 附录 C。

重建时在项目根目录使用具有项目写权限的 Python 环境，运行 `python docs/b_textbook/build_pdf.py`；用 Poppler `pdftoppm` 渲染全部页到 `tmp/pdfs/b_textbook/`，运行 `python docs/b_textbook/qa_pdf.py`，再逐页检查排版。此次已渲染 33/33 页并检查缩略图及代表页；中文、图示、表格和代码未见截断。此目录不是训练断点入口。
