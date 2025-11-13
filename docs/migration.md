# 架构迁移说明

## 目标
- 依据 todo.md 设计，引入分层模块化 `PyCoCoContactLib` 包。
- 保持现有业务逻辑不变，所有调用通过 `PyCoCoContactLib` 标准接口。

## 映射关系
- core.mesh → ss_compare.Mesh（封装为 dataclass Mesh）
- core.bvh → ss_compare.build_bvh / traverse_bvhs
- detection.* → 复用 ss_compare 的候选/精检测/活跃面方法
- force.* → 复用 contact_force_calculator 的赫兹样模型与积分
- io.obj_loader → 复用 obj_contact_demo.load_obj_as_mesh
- utils.transforms → 复用 obj_contact_demo_with_pose 位姿工具
- visualization.* → 复用 obj_contact_demo 可视化接口
- api.contact_solver → 统一入口，端到端封装

## 迁移策略
1. 不修改原始脚本，实现适配层包化封装。
2. 新增 `tests/` 验证端到端功能完整性。
3. 生成迁移日志与回滚方案文档。

## CI/CD
- 当前仓库无现有 CI 定义；新增 `tests/` 后可直接集成到常见流水线。
