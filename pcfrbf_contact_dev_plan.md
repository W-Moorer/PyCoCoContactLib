
# 基于 PCFRBF 的接触检测集成开发计划（面向当前 rcsim 框架）

> 目标：在 **保持现有 AABB-BVH 粗检测不变** 的前提下，将上一版设计的 **PCFRBF 窄阶段接触检测** 集成到当前 `rcsim` 框架中，并通过循序渐进的实现与验证，最终在动力学仿真中稳定使用。

---

## 总体路线图概览

按阶段推进，每一阶段都包含两部分：**实现内容** + **验证方式**：

1. **阶段 0：熟悉与基线验证**  
   跑通现有框架，确认当前接触检测与动力学是“可工作的”基线。

2. **阶段 1：抽象“局部隐式面片接口” + 平面 Patch 实现**  
   先用“平面隐式函数”实现框架，验证接口设计合理，而不立刻引入 PCFRBF 的复杂性。

3. **阶段 2：实现并验证“点–局部隐式面片”的迭代投影算法（窄阶段内核）**  
   在单个三角面片上，利用平面 Patch 验证迭代投影流程是正确、收敛的。

4. **阶段 3：将“隐式面片投影”集成到现有 `MeshPairContactDetector` 中（仍然用平面 Patch）**  
   在现有三角形接触框架中引入“采样点 → 局部隐式曲面投影”，并与旧的 tri/tri 距离结果对比。

5. **阶段 4：替换平面 Patch 为 PCFRBF Patch，实现 PCFRBF 几何评估与投影**  
   在接口保持不变的前提下，实现 PCFRBF 从数据到 `phi`/`grad` 评估，并插入窄阶段。

6. **阶段 5：物理级别验证与性能调优**  
   在典型场景下比较接触稳定性、能量表现、性能，与旧算法对比。

下面按阶段详细展开。

---

## 阶段 0：熟悉现有框架与基线确认

### 0.1 阅读关键模块

重点阅读如下文件（你已经有一部分认知，但建议系统化整理）：

- `rcsim/contact/aabb_bvh.py`  
  - `Aabb` / `AabbWithData` / `BvhTree` 的接口，尤其 `build`、`find_collision`。
- `rcsim/contact/mesh.py`  
  - `RapidMesh` 的数据结构：`vertices`、`triangles`、`bvh`、`tri_size`、`center_local`、`radius`。  
  - `get_triangle_nodes(tri_index, pose)` 的接口是窄阶段的关键入口。
- `rcsim/contact/tri_geom.py`  
  - 现有三角形几何例程：`triangle_distance`、`TriangleDistance2`、`triTriIntersect`。
- `rcsim/contact/detector.py`  
  - `MeshPairContactDetector`：  
    - `query_manifold(...)`：整套接触检测入口。  
    - `_build_candidates_from_pairs(...)`：从 BVH 粗检测得到的 tri 对生成候选接触点。  
    - `_project_prev_manifold(...)`：基于前一帧流形的“时间连续性投影”。  
    - `_reduce_candidates_to_manifold(...)`：从候选点筛选/合并生成最终 `ContactManifold`。  
    - `closest(...)`：全局最近距离查询。
- `rcsim/physics/world.py`  
  - `ContactPair` / `World.compute_contact_forces`：`ContactManifold` 的 `cp.n` / `cp.phi` 如何进入力学计算。

> 目的：清晰画出 “BVH → tri 对 → 窄阶段 → ContactManifold → 力学” 的调用链。

### 0.2 运行一个最小示例场景

- 自己写一个小脚本（例如 `examples/minimal_world.py`）：
  - 读入两个简单 mesh（例如两个立方体），各放置在略有重叠或接触的位置；
  - 构造 `RigidMeshBody`，构建 `World`，调用一步 `step`；
  - 打印出生成的 `ContactManifold`：
    - `len(manifold.points)`
    - 每个 `cp.phi`, `cp.n`, `cp.pA_world`, `cp.pB_world`
- 验证：
  - 接触点数量在合理范围；
  - `phi < 0` 时才有接触力；
  - 接触法向大致与几何直观一致。

> 这一阶段的目标：得到一个“旧算法的可靠基线”，后续所有改动都可以与此对比。

---

## 阶段 1：抽象“局部隐式面片接口” + 平面 Patch 实现

在引入 PCFRBF 之前，先建立一个通用接口，使得不同局部隐式表示（平面 / PCFRBF）可以互换。

### 1.1 设计接口：局部隐式面片 LocalImplicitPatch

在 `rcsim/contact/pcfrbf_patches.py`（或 `rcsim/geom/local_implicit.py`）中新建接口，例如：

```python
import numpy as np
from dataclasses import dataclass

@dataclass
class LocalImplicitPatch:
    """局部隐式曲面 patch 的通用接口：定义在三角形局部坐标中"""
    # 如需，可记录三角顶点的局部坐标、局部坐标系等

    def phi(self, x_local: np.ndarray) -> float:
        raise NotImplementedError

    def grad_phi(self, x_local: np.ndarray) -> np.ndarray:
        raise NotImplementedError
```

注意设计点：

- `x_local` 使用三角形的局部坐标系，便于不同 patch（平面 / PCFRBF）共享同一坐标描述；
- 接口尽量简洁，只暴露 `phi` 与 `grad_phi`。

### 1.2 实现平面 Patch：PlanarTrianglePatch

在同一文件中实现一个“基于三角面片平面的隐式函数”：

- 三角面片顶点（局部）为 v0, v1, v2；
- 法向为

  n = ((v1 - v0) × (v2 - v0)) / ||(v1 - v0) × (v2 - v0)||

- 平面隐式函数可定义为

  phi_plane(x) = (x - v0) · n

实现示意：

```python
@dataclass
class PlanarTrianglePatch(LocalImplicitPatch):
    v0: np.ndarray
    v1: np.ndarray
    v2: np.ndarray
    n: np.ndarray  # 单位法向

    def phi(self, x_local: np.ndarray) -> float:
        return float(np.dot(x_local - self.v0, self.n))

    def grad_phi(self, x_local: np.ndarray) -> np.ndarray:
        # 平面上的梯度恒为 n
        return self.n.copy()
```

### 1.3 验证：平面 Patch 的正确性

单元测试（建议用 `pytest`，例如 `tests/test_planar_patch.py`）：

1. 在局部坐标系下构造一个标准三角形：
   - v0 = (0,0,0)
   - v1 = (1,0,0)
   - v2 = (0,1,0)
   - 法向 n = (0,0,1)
2. 验证：
   - 三角形平面上的点（如重心） phi ≈ 0；
   - 平面上方/下方一点 (0,0,h) 的 phi ≈ h；
   - `grad_phi` 始终等于 (0,0,1)。

通过这些测试，保证接口定义与实现是自洽的。

---

## 阶段 2：实现“点–局部隐式面片”的迭代投影算法（以平面 Patch 验证）

这一阶段在单个面片上实现并验证“迭代投影”内核逻辑，不考虑整个 mesh 或动力学，只验证几何算法正确性。

### 2.1 实现投影函数（局部）

在 `pcfrbf_patches.py` 中实现一个通用的投影函数：

```python
from typing import List, Tuple

def project_point_to_patch(
    x_world: np.ndarray,
    tri_vertices_world: List[np.ndarray],
    patch: LocalImplicitPatch,
    max_iters: int = 5,
    eps_phi: float = 1e-8,
    eps_y: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    输入：世界系下点 x_world，三角面片顶点 tri_vertices_world，和对应的局部隐式 patch。
    输出：投影点 y_world，法向 n_world，有符号距离 g。
    """
    ...
```

实现步骤（与之前设计一致）：

1. 线性三角形最近点求初值（可以直接复用/移植 `tri_geom._point_triangle_distance` 的逻辑，或重新写一个专门的版本）；
2. 将初值转换到局部坐标系（可选，视整体设计是“先定义局部系再变回世界”还是直接在世界系上迭代而定）；
3. 每次迭代：

   y_{k+1} = y_k - phi(y_k) / ||grad_phi(y_k)||^2 * grad_phi(y_k)

4. 域限制：把 `y_{k+1}` 投回三角平面，计算重心坐标，若越界则 clamp 到边/顶点；
5. 检查收敛条件：
   - |phi| < eps_phi 或
   - ||y_{k+1} - y_k|| < eps_y 或
   - 达到最大迭代次数；
6. 最终计算世界系法向和有符号距离：

   g = (x_world - y_world) · n_world

对于平面 Patch，迭代实际上在 1～2 步内就能收敛到解析解。

### 2.2 验证：与解析平面投影的一致性

在 `tests/test_project_point_to_patch.py` 中：

1. 构造一个简单平面三角形（同 1.3）；
2. 生成一批随机点：
   - 一部分在面片附近；
   - 一部分稍远；
3. 对每个点：
   - 用解析几何求最近点（线性三角形最近点）和法向距离（现有 `tri_geom` 已可作为参考）；
   - 用 `project_point_to_patch` 得到 `y_world`, `n_world`, `g`；
   - 验证：
     - 最近点误差 ||y_new - y_tri|| 在容差内；
     - 距离误差 |g_new - d_tri| 在容差内；
     - 法向方向基本一致（点乘接近 1）。

通过这一阶段，保证 “点–局部隐式面片投影” 算法逻辑正确，且与传统几何结果一致。

---

## 阶段 3：集成到 MeshPairContactDetector（仍使用平面 Patch）

本阶段把阶段 2 的局部投影内核，融入现有的 `MeshPairContactDetector`，但为了降低复杂度，仍然使用平面 Patch 作为隐式表示。

### 3.1 为每个三角面片附加 Patch 数据

在 `RapidMesh` 中添加用于存储 / 构造 patch 的接口。例如在 `rcsim/contact/mesh.py` 中：

- 在 `RapidMesh.__init__` 中为每个三角面片预计算：
  - 局部坐标系（如平面基向量 + 法向）；
  - 对应的 `PlanarTrianglePatch` 实例；
- 暴露一个接口，例如：

  ```python
  class RapidMesh:
      ...
      def get_patch(self, tri_index: int) -> LocalImplicitPatch:
          return self.patches[tri_index]
  ```

在这一阶段，`patches[i]` 全部是 `PlanarTrianglePatch`。

### 3.2 修改 _build_candidates_from_pairs 使用“点–Patch 投影”

在 `MeshPairContactDetector._build_candidates_from_pairs` 中：

- 目前的流程大致是：
  1. 对 BVH 给出的 tri 对 `(ia, ib)`：
     - 用 `triTriIntersect` 生成穿透点；
     - 用 `TriangleDistance2` 生成最近分离点；
  2. 把这些结果转成 `ContactPoint`，加入 `candidates`。

- 改造后（窄阶段的核心变化）：
  1. 对每个 tri 对 `(ia, ib)`：
     - 从 `ia` 上采样若干点：顶点 + 边中点 +（可选）重心，得到 {x_p^A}；
     - 从 `ib` 上采样若干点，得到 {x_q^B}；
  2. 调用 `project_point_to_patch`：
     - 对每个 `x_p^A` 投影到 `ib` 的 patch；
     - 对每个 `x_q^B` 投影到 `ia` 的 patch；
  3. 对每个成功的投影，构造 `ContactPoint`：
     - `pA_world` / `pB_world`：分别是原点和投影点；
     - `n`：来自 patch 的世界系法向（方向可统一规定为从 B 指向 A 或反之）；
     - `phi`：有符号距离（注意符号约定与 `world.py` 中的使用保持一致，接触时应为负）。

注意事项：

- 在引入新算法的初期，可以保留原有 tri/tri 算法，同时添加一个开关：
  - 如 `MeshPairContactDetector` 增加一个 flag：`use_implicit_narrow_phase: bool`；
  - 便于对比新旧算法的输出与性能。

### 3.3 验证：与旧 tri/tri 接触的结果对比

写一个对比脚本（例如 `tools/compare_contact_detectors.py`）：

1. 构建若干静态场景（如两个立方体、球与平面、斜放的棱柱与平面等），对每个场景：
   - 记录物体的 pose；
   - 分别调用：
     - 旧版 `MeshPairContactDetector`（tri/tri）；
     - 新版 `MeshPairContactDetector`（平面 Patch 窄阶段）；
   - 比较：
     - 接触点数量；
     - 各 `cp.phi` 的符号与大小；
     - 各 `cp.n` 的方向（与旧版 dot product 接近 1 即可）。

2. 允许存在一定差异，但要确保：
   - 不会漏检测明显的接触；
   - 不会产生大量明显虚假接触点（远离真实接触区域）；
   - 在简单几何（平面、盒子）下，新算法与旧算法结果高度一致。

3. 将测试脚本集成到简单的 `pytest` 或 `Makefile` 流程中，以便回归测试。

到此为止：不涉及 PCFRBF，仅用平面 Patch，目标是把“隐式窄阶段管线”稳定嵌入现有框架。

---

## 阶段 4：引入 PCFRBF Patch 并替换平面 Patch

在上一阶段接口已经稳定的前提下，本阶段的工作是：实现真正的 PCFRBF Patch，替换（或并存）平面 Patch。

### 4.1 设计 PcfrbfPatch 数据结构与构造流程

在 `pcfrbf_patches.py` 中实现：

```python
from dataclasses import dataclass

@dataclass
class PcfrbfPatch(LocalImplicitPatch):
    # RBF 相关数据，如中心、权重、核参数等
    centers: np.ndarray      # (M, 3) 或在局部坐标中
    weights: np.ndarray      # (M,)
    kernel_param: float
    v0: np.ndarray           # 参考点（例如三角一个顶点）
    local_frame: np.ndarray  # 3x3 局部坐标系基
    # 如需，可加入边界控制参数等

    def phi(self, x_local: np.ndarray) -> float:
        ...

    def grad_phi(self, x_local: np.ndarray) -> np.ndarray:
        ...
```

构造流程建议：

- 将 PCFRBF 拟合与系数求解放在离线预处理流程中（独立脚本），输出一个数据文件，例如：
  - `my_mesh.pcfrbf.json` 或 `npz`，存储每个三角面片的 patch 参数；
- 在 `RapidMesh` 或某个 `PcfrbfRapidMesh` 的构造函数中，从该数据文件加载对应 triangle 的 patch 参数，组装为 `PcfrbfPatch`。

若前期还未完成 PCFRBF 拟合代码，可以先用“手工构造的简单 RBF”（如基于一个平面 + 少量扰动）做最小可用版本，后续再替换为真实拟合结果。

### 4.2 实现 phi 与 grad_phi 的数值评估

根据 PCFRBF 的形式，实现：

- phi(x) = sum_i w_i * phi_kernel(||x - c_i||)，其中 phi_kernel(r) 为选定的 RBF 核；
- grad_phi(x) = sum_i w_i * phi_kernel'(r) * (x - c_i) / r，或根据 curl-free 形式写出解析表达。

注意数值稳定性：

- 当 r 很小时，避免除以 0；
- 可以用核函数的泰勒展开或直接对 r 做截断。

### 4.3 单 Patch 层面的精度验证

新建测试（如 `tests/test_pcfrbf_patch_accuracy.py`）：

1. 对于某个面片 patch：
   - 在其局部邻域内采样一批点；
   - 对于“真实表面”的点（如原始 mesh 上的点或高精度参考），验证：
     - |phi| 足够小（接近 0）；
     - `grad_phi` 方向与原始 mesh 法向夹角小于若干度（如 5°）。
2. 对于离表面一定距离的点：
   - 验证 phi 的符号与几何关系一致；
   - 验证 `project_point_to_patch` 收敛且投影点位于合理区域（例如与线性三角投影结果差距不大）。

### 4.4 将 RapidMesh 中 Patch 切换为 PcfrbfPatch

在 `RapidMesh` 中增加配置：

- 若存在对应的 PCFRBF 数据文件，则为每个 triangle 构建 `PcfrbfPatch`，否则回退到 `PlanarTrianglePatch`；
- 或提供两个类：
  - `RapidMesh`（平面 Patch）；
  - `PcfrbfRapidMesh`（PCFRBF Patch）；

并在场景构建处决定用哪个 mesh 类型。

### 4.5 集成验证：对比平面 Patch 与 PCFRBF Patch

重复阶段 3 的对比测试：

- 对同一场景：
  - 用平面 Patch 的接触结果作为 baseline；
  - 用 PCFRBF Patch 的接触结果对比：
    - 接触点位置变化是否合理（PCFRBF 会更贴近真实曲面，与平面相比的偏差在可接受范围内）；
    - 法向是否更加平滑（检查邻近接触点法向的变化）；
    - 不出现明显的“穿透翻转”或大面积错判。

---

## 阶段 5：物理级验证与性能调优

最后，在完整的物理仿真层面验证 PCFRBF 接触检测的实际效果。

### 5.1 典型场景测试

为下列场景分别创建脚本（例如 `examples/pcfrbf_box_stack.py`, `examples/pcfrbf_gear_contact.py` 等）：

1. 箱体堆叠 / 物块堆叠：
   - 多个立方体堆叠，测试接触稳定性、穿透深度是否保持小且平稳；
2. 斜面滑动：
   - 一个盒子沿斜面滑动，检查接触法向与摩擦行为是否平滑；
3. 高曲率物体接触（如果有 PCFRBF 建模的复杂曲面）：
   - 如球面、齿轮、自由曲面等，重点观察：
     - 接触法向是否随接触点移动而平滑变化；
     - 是否出现突然的法向翻转或跳变。

对于每个场景，记录：

- 接触点数量随时间的变化；
- 穿透深度 -phi 的统计量（最大值、均值）；
- 能量行为（若有能量统计模块，可记录系统总能量随时间的变化）。

### 5.2 性能评估与优化

在 `rcsim.io.perf` 的帮助下，测量：

- 粗检测时间：`bvh.find_collision`；
- 新窄阶段时间：`detector.build_candidates` + 投影迭代时间；
- 与旧 tri/tri 实现相比的性能开销。

可能的优化方向：

- 减少每个 tri 对的采样点数量，找到精度与速度的平衡点；
- 利用前一帧缓存的投影结果作为初值，减少迭代次数；
- 在 PCFRBF 的 `phi` / `grad_phi` 计算中优化向量化与缓存中间量。

### 5.3 回归测试与配置管理

- 为 PCFRBF 窄阶段提供一个可配置的开关（例如在 `World` 或场景配置中），支持：
  - `mode = "tri_tri"`（旧算法）
  - `mode = "plane_implicit"`
  - `mode = "pcfrbf_implicit"`
- 为关键场景建立回归测试：
  - 确保改动不会破坏已有稳定行为；
  - 可以在 CI 中定期跑小规模测试，检查接触数量与穿透深度不出现突变。

---

## 附录：建议的实现顺序一览

可用作日常开发 checklist：

1. 阶段 0
   - [ ] 阅读并整理现有接触与物理框架调用关系图  
   - [ ] 写最小示例脚本，确认旧版接触检测正常工作

2. 阶段 1
   - [ ] 定义 `LocalImplicitPatch` 接口  
   - [ ] 实现 `PlanarTrianglePatch`  
   - [ ] 为平面 Patch 写单元测试（phi、grad_phi 正确）

3. 阶段 2
   - [ ] 实现 `project_point_to_patch`（迭代投影）  
   - [ ] 编写平面 Patch 上的投影单元测试，与解析最近点/距离对比

4. 阶段 3
   - [ ] 在 `RapidMesh` 中为每个 triangle 预生成 Patch（先用平面 Patch）  
   - [ ] 修改 `MeshPairContactDetector._build_candidates_from_pairs`，加入“采样点 → Patch 投影”逻辑  
   - [ ] 写对比脚本，比较旧 tri/tri 与新平面 Patch 窄阶段的接触结果

5. 阶段 4
   - [ ] 定义 `PcfrbfPatch` 数据结构  
   - [ ] 实现 PCFRBF 的 phi / grad_phi 评估  
   - [ ] 为单 Patch 写精度测试（phi 逼近 0、法向与真实法向一致）  
   - [ ] 在 `RapidMesh` 中加载 PCFRBF 数据，切换 Patch 类型  
   - [ ] 对比平面 Patch 与 PCFRBF Patch 的接触结果

6. 阶段 5
   - [ ] 构建多个典型场景，观测接触稳定性与数值表现  
   - [ ] 使用 `perf` 评估性能并进行必要优化  
   - [ ] 加入回归测试与配置开关，完成整体集成

---

该开发计划文件可以直接作为你在 rcsim 中集成 PCFRBF 接触检测的路线图使用，也便于后续在代码仓库中保存为 `docs/pcfrbf_contact_dev_plan.md` 以供长期维护与迭代。
