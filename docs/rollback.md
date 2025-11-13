# 回滚方案

- 保留原始脚本：`ss_compare.py`、`contact_force_calculator.py` 等均未更改。
- 回滚步骤：
  1. 停用通过 `PyCoCoContactLib` 的入口调用。
  2. 直接恢复旧的脚本式调用（示例：`run_demo.py` 调用 `obj_contact_demo_with_pose`）。
  3. 如需删除 `PyCoCoContactLib`，仅移除包目录，不影响原脚本运行。
