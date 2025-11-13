<div align="center">
  <pre style="font-family: 'Courier New', monospace; 
              line-height: 1.2; 
              white-space: pre-wrap;
              display: inline-block;
              padding: 10px;
              border-radius: 4px;
              border: 1px solid #3b0aceff;">
$$$$$$$\             $$$$$$\             $$$$$$\                                              
$$  __$$\           $$  __$$\           $$  __$$\                                             
$$ |  $$ |$$\   $$\ $$ /  \__| $$$$$$\  $$ /  \__| $$$$$$\                                    
$$$$$$$  |$$ |  $$ |$$ |      $$  __$$\ $$ |      $$  __$$\                                   
$$  ____/ $$ |  $$ |$$ |      $$ /  $$ |$$ |      $$ /  $$ |                                  
$$ |      $$ |  $$ |$$ |  $$\ $$ |  $$ |$$ |  $$\ $$ |  $$ |                                  
$$ |      \$$$$$$$ |\$$$$$$  |\$$$$$$  |\$$$$$$  |\$$$$$$  |                                  
\__|       \____$$ | \______/  \______/  \______/  \______/                                   
          $$\   $$ |                                                                          
          \$$$$$$  |                                                                          
           \______/                                                                           
 $$$$$$\                        $$\                           $$\     $$\       $$\ $$\       
$$  __$$\                       $$ |                          $$ |    $$ |      \__|$$ |      
$$ /  \__| $$$$$$\  $$$$$$$\  $$$$$$\    $$$$$$\   $$$$$$$\ $$$$$$\   $$ |      $$\ $$$$$$$\  
$$ |      $$  __$$\ $$  __$$\ \_$$  _|   \____$$\ $$  _____|\_$$  _|  $$ |      $$ |$$  __$$\ 
$$ |      $$ /  $$ |$$ |  $$ |  $$ |     $$$$$$$ |$$ /        $$ |    $$ |      $$ |$$ |  $$ |
$$ |  $$\ $$ |  $$ |$$ |  $$ |  $$ |$$\ $$  __$$ |$$ |        $$ |$$\ $$ |      $$ |$$ |  $$ |
\$$$$$$  |\$$$$$$  |$$ |  $$ |  \$$$$  |\$$$$$$$ |\$$$$$$$\   \$$$$  |$$$$$$$$\ $$ |$$$$$$$  |
 \______/  \______/ \__|  \__|   \____/  \_______| \_______|   \____/ \________|\__|\_______/ 
                                                                                              
                                                                                              
                                                                                              
  </pre>
</div>


# PyCoCoContactLib

轻量分层的接触检测与接触力计算库。提供 Mesh/BVH、粗-精检测、主动面片、Dunavant 积分、Hertz-like 力学模型、OBJ IO、PyVista 可视化与统一求解器接口。

## 项目背景
- 基于现有接触几何与力学实现，重构为分层模块化库，统一入口、清晰边界、易扩展。
- 保持原有功能完整性，业务逻辑不变；通过标准接口封装与调用，实现无缝迁移与分发。

## 架构设计
```
PyCoCoContactLib/
├── __init__.py                 # 统一入口，导出核心API
├── core/                       # 核心数据结构与基础算法
│   ├── __init__.py
│   ├── mesh.py                 # Mesh类 + 基础几何操作
│   ├── bvh.py                  # BVH构建与遍历
│   ├── geometry.py             # 三角形几何工具
│   └── sphere.py               # 基于正二十面体的球体生成
├── detection/                  # 接触检测模块
│   ├── __init__.py
│   ├── broad_phase.py          # BVH粗检测
│   ├── narrow_phase.py         # Möller精检测
│   ├── active_faces.py         # 主动面片提取
│   └── sampling.py             # Dunavant采样抽取接触点
├── force/                      # 接触力计算模块
│   ├── __init__.py
│   ├── models.py               # Hertz-like模型定义
│   ├── integrator.py           # 分布式积分器
│   └── calculator.py           # 统一计算接口
├── io/                         # 文件IO模块
│   ├── __init__.py
│   ├── obj_loader.py           # OBJ加载器
│   └── mesh_io.py              # 通用网格IO
├── visualization/              # 可视化模块（可选）
│   ├── __init__.py
│   ├── pyvista_backend.py      # PyVista渲染器
│   └── contact_plotter.py      # 接触可视化工具
├── utils/                      # 工具模块
│   ├── __init__.py
│   ├── transforms.py           # 位姿变换（四元数/矩阵）
│   └── validation.py           # 参数校验
├── api/                        # 高级API
│   ├── __init__.py
│   ├── contact_solver.py       # 统一求解器
│   └── demo_utils.py           # 示例工具
└── quadrature.py               # Dunavant7 等求积规则
```

## 关键模块设计（概览）
- `core.mesh.Mesh`：`V` 顶点、`F` 面片；提供质心和内部网格桥接。
- `core.bvh`：中位数分割构建、AABB 交叠判定、双树遍历候选对。
- `detection.broad_phase`：基于 BVH 返回潜在接触的三角形对。
- `detection.narrow_phase`：Möller 精确三角形相交判定。
- `detection.active_faces`：候选 ∪ 精检测 + 最近点签名距离，提取主动面片。
- `detection.sampling`：Dunavant 采样在主动面片上抽取穿透点，插值法向并计算投影面积权重。
- `force.models.HertzModel`：`F = k * δ^n + d * δ_dot` 单点法向力模型。
- `force.integrator.DistributedIntegrator`：分布式接触点积分，输出总力与压力分布。
- `io.obj_loader.load_obj`：极简 OBJ 解析（仅 v/f，多边形扇形三角化）。
- `utils.transforms`：`quat_wxyz_to_rotmat`、`apply_pose_to_mesh` 刚体位姿。
- `visualization.pyvista_backend`：网格与接触压力渲染，合力箭头随模型尺度和力大小动态缩放。
- `api.contact_solver.ContactSolver`：统一入口，内部默认 `HertzModel + MedianSplitBVH + Dunavant7`。

## 功能特性
- 模块解耦：核心算法与 IO/可视化分离；任意组件可替换（BVH策略、力学模型等）。
- 可扩展性：通过继承基类扩展模型/算法；插件式架构支持第三方扩展。
- 易用性：高级 API 隐藏复杂实现；合理默认参数降低使用门槛。
- 性能优化：关键路径可引入 Numba 加速（可选）；内存池策略（规划中）。
- 类型安全：全面类型注解与运行时参数校验。

## 安装
- 开发（可编辑安装）：`python -m pip install -e .`
- 运行环境（标准安装）：`python -m pip install .`

## 使用说明
```python
from PyCoCoContactLib import ContactSolver, Mesh, load_obj

# 加载网格（示例模型位于 models/）
mesh_a = load_obj("models/ballMesh.obj")
mesh_b = load_obj("models/cubeMesh.obj")

# 创建求解器（使用默认参数）
solver = ContactSolver()

# 计算接触力
result = solver.compute(mesh_a, mesh_b, skin=0.0)
print("总接触力:", result.total_force)
print("最大穿透深度:", result.max_penetration)
```

自定义模型：
```python
from PyCoCoContactLib.force.models import HertzModel
from PyCoCoContactLib.api.contact_solver import ContactSolver

custom_model = HertzModel(k=2e6, n=1.8, d=5e3)
solver = ContactSolver(model=custom_model)
result = solver.compute(mesh_a, mesh_b)
```

可视化：
```python
from PyCoCoContactLib.visualization.pyvista_backend import visualize_contact
visualize_contact(mesh_a, mesh_b, result.contact_points, result.pressures, result.total_force)
```

位姿应用：
```python
from PyCoCoContactLib.utils.transforms import apply_pose_to_mesh
mesh_a_world = apply_pose_to_mesh(mesh_a, origin=(0,0,0.1), quat_wxyz=(1,0,0,0))
```

## 开发环境要求
- Python `>=3.8`
- 运行依赖：`numpy>=1.20`
- 可视化（可选）：`pyvista`、`matplotlib`
- 开发工具（可选）：`pytest`、`black`、`mypy`

## 开发与测试
- 运行测试：`python -m unittest discover -s tests -p "test*.py"`
- 代码风格：PEP8，类型标注优先，避免在代码中加入注释

## 构建与发布
- 构建：`python -m pip install build twine`，`python -m build`
- 发布：`python -m twine upload dist/*`
- GitHub Actions：推送带版本标签后自动发布（需配置仓库机密 `PYPI_API_TOKEN`）

## CI/CD 与质量要求
- 保持 CI/CD 流程可用，集成单元与集成测试。
- 所有测试 100% 通过，核心功能在新架构下运行正常。
- 性能指标不劣于原框架实现（适配层无额外复杂度）。

## 未来计划（TODO）
- 引入 Numba/JIT 加速关键几何与遍历路径。
- 增加多种 BVH 构建策略与参数化选择。
- 扩展可视化交互（选择接触区域、压力等值线）。
- 内存池与批量流水优化，减少分配开销。
- 更完善的类型校验与错误报告。
- 提供更多示例与教程文档。

## 版本与兼容
- 语义化版本：MAJOR.MINOR.PATCH
- Python 版本：`>=3.8`

## 许可
MIT
