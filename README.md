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

一个用于三角网格快速接触与最近距离估算的轻量级 Python 库与演示。仓库已重构为分层的 `rcsim/` 包，包含接触几何、物理世界/刚体与 Velocity-Verlet 时间积分器，并提供演示脚本与结果导出。

## 仓库结构
```
PyCoCoContactLib/
├── demos/
│   └── demo_drop.py              # 自由落体接触演示（CSV/日志/绘图）
├── models/
│   └── ballMesh.obj              # 示例网格（球）
├── outputs/                      # 运行后生成的结果目录（统一在仓库根下）
│   └── <name>_<mesh>_<时间>/     # CSV/日志/绘图与 perf_metrics.csv
├── post/
│   └── plot_perf_pie.py          # 性能占比饼图后处理脚本
├── rcsim/
│   ├── contact/                  # 接触几何与检测
│   │   ├── aabb_bvh.py           # AABB/BVH 加速结构
│   │   ├── contact_types.py      # 接触点/流形与类型
│   │   ├── detector.py           # Mesh 对检测与流形查询
│   │   ├── mesh.py               # OBJ 加载与 RapidMesh
│   │   └── tri_geom.py           # 三角形相交/最近距离
│   ├── physics/                  # 刚体/世界与积分器
│   │   ├── rigid_bodies.py       # RigidBody / RigidMeshBody
│   │   ├── world.py              # ContactWorld（重力+惩罚弹簧+阻尼）
│   │   └── integrators/
│   │       └── verlet.py         # Velocity‑Verlet 与多子步仿真
│   └── io/
│       ├── perf.py               # 轻量性能记录器（CSV 输出 perf_metrics.csv）
│       ├── recorders.py          # CSV 记录器示例
│       └── __init__.py           # 导出 csv_recorder
├── LICENSE
└── README.md
```

## 环境与依赖
- Python 3.9+
- 依赖：`numpy`（可选：`matplotlib` 用于绘图）

安装依赖：
```
pip install numpy matplotlib
```

## 快速上手
使用演示脚本运行自由落体接触并生成 CSV 与日志。

- 无阻尼（对应原始 undamped 演示）：
```
python -m demos.demo_drop --mesh models/ballMesh.obj --time 8.0 --dt 0.01 --sub 0.001 --out mesh_drop_results_rk4.csv --log rcsim_output_rk4.log --progress 0.01 --verbose --integrator rk4 --video --fps 120 --stride 1
```

- 阻尼/半波阻尼（对应原始 damped 演示）：
```
python -m demos.demo_drop --mesh models/ballMesh.obj --time 8.0 --dt 0.01 --sub 0.0005 --damp 100.0 --half_wave --out mesh_drop_results_damped_half_wave.csv --log rcsim_output_damped_half_wave.log --progress 1.0 --verbose --video --fps 120 --stride 1 
```

运行成功后，所有结果统一写入仓库根下的 `outputs/` 目录，路径格式：
`outputs/<name>_<meshStem>_<YYYYMMDD_HHMMSS>/`，其中 `<name>` 可通过 `--name` 参数自定义（默认 `output`）。目录中包含 CSV、日志、绘图 PNG 与 `perf_metrics.csv`。
替换网格时，将 `--mesh` 指向你的 OBJ 文件路径。

生成图像（仅从 CSV 绘制掉落物体曲线）：
```
python -m demos.demo_drop --mesh models/ballMesh.obj ... --plot
```
或直接调用：
```
python -c "from demos.demo_drop import plot_results; plot_results('models/ballMesh/mesh_drop_results_rapid_verlet.csv','', 'models/ballMesh')"
```

## API 概览
- 最近距离与法线估算：
```python
import numpy as np
from rcsim.contact import RapidContactDetectionLib

rcd = RapidContactDetectionLib('models/ballMesh.obj')
poseA = (np.array([0.0, 0.0, 205.0]), np.eye(3))
poseB = (np.array([0.0, 0.0,   0.0]), np.eye(3))
dist, n = rcd.closest(poseA, poseB)
print('closest distance =', dist)
print('normal =', n)
```

- 统一世界与仿真接口：
```python
from rcsim import RapidMesh, RigidMeshBody, ContactWorld, simulate_verlet

mesh = RapidMesh.from_obj('models/ballMesh.obj')
bodies = [
    RigidMeshBody(mesh, mass=1.0, position=[0.0,0.0,0.0],   velocity=[0.0,0.0,0.0], is_static=True),
    RigidMeshBody(mesh, mass=1.0, position=[0.0,0.0,205.0], velocity=[0.0,0.0,0.0], is_static=False),
]
world = ContactWorld(bodies, g=-9.8, k_contact=1e5, c_damp=0.0)
world.build_all_pairs()

def on_frame(frame, t, w):
    ball = w.bodies[1].body
    print(frame, t, ball.position[2])

simulate_verlet(world, t_end=2.0, dt_frame=0.01, dt_sub=1e-3, on_frame=on_frame)
```

## 结果输出与性能统计
- 输出目录：所有运行结果统一保存到仓库根目录 `outputs/` 下，目录命名为 `outputs/<name>_<meshStem>_<YYYYMMDD_HHMMSS>/`。
  - 通过参数 `--name` 自定义 `<name>` 前缀，默认 `output`。
  - 目录内包含：`<out>.csv`、日志 `<log>.log`、各类绘图 PNG 与 `perf_metrics.csv`（性能摘要）。
- 性能记录：内置轻量记录器会统计关键模块总耗时、次数与占比并写出 `perf_metrics.csv`，字段包括 `key,total_sec,count,avg_ms,percent,algorithm,t_end,dt_frame,dt_sub,bodies`。
 - 性能记录：内置轻量记录器统计“排除嵌套”的模块耗时（exclusive）并写出 `perf_metrics.csv`：
   - 字段：`key,exclusive_sec,inclusive_sec,count,avg_ms,exclusive_percent,algorithm,t_end,dt_frame,dt_sub,bodies`
   - 最后一行 `TOTAL` 给出总耗时（用于图例标题展示），不参与饼图占比计算。

### 性能占比饼图（后处理）
- 脚本位置：`post/plot_perf_pie.py`
- 用法示例：
  - 处理 `outputs/` 下的文件并指定输出路径：
    ```
    python post/plot_perf_pie.py --perf ".\outputs\output_ballMesh_YYYYMMDD_HHMMSS\perf_metrics.csv" --out ".\outputs\perf_pie.png"
    ```
  - 处理任意现有文件（例如旧示例目录），使用默认输出到同目录：
    ```
    python post/plot_perf_pie.py --perf ".\models\ballMesh\viz_20251118_231655\perf_metrics.csv"
    ```
- 绘图说明：
  - 标题采用 `perf_metrics.csv` 所在文件夹名；字体使用 `Times New Roman`。
  - 饼图内部仅显示较大扇区的百分比；右侧图例列出模块名、百分比与耗时，并在图例标题处显示总耗时（来自 CSV 的 `TOTAL` 行）。

## 框架更新摘要
- 统一输出位置：由 `models/<mesh>/...` 改为仓库根 `outputs/`，路径格式 `outputs/<name>_<meshStem>_<时间>/`。
- 新参数：`demos/demo_drop.py` 支持 `--name` 自定义输出目录前缀（默认 `output`）。
- 性能统计：新增 `rcsim/io/perf.py`，在积分器、世界与检测器关键路径埋点，自动写出 `perf_metrics.csv`。
- 后处理：新增 `post/plot_perf_pie.py`，生成性能占比饼图，支持自定义输出路径或默认写入源目录。

## 数值收敛与步长选择（补充）
以下为自由落体接触算例的收敛性观察与建议，基于多组时间步比较：

- 参考解与主结论
  - 以 `Δt = 1e-5` 作为参考解；在工程关注的“第一次接触/半个波形”阶段，`Δt = 1e-4、2.5e-4、5e-4` 的位移时间历程与参考解几乎重合，误差量级约 `5×10⁻³%`，可认为已收敛。
  - 若观察 0–3 s 的主响应，`Δt = 1e-4、2.5e-4、5e-4（RK4）` 对参考解的均方根相对误差 ≲ `0.02%`，可视为“工程上非常干净”的收敛。

- 长尾阶段与相位/能量误差
  - 在 5–8 s 的衰减尾段，不同步长间存在相位/能量的累积差异；某些时刻位移差异可达 `0.6–0.85`（相对当天的微小振幅约 `10–15%`）。这是尾部小振幅阶段的相位漂移，更敏感于步长与数值阻尼。
  - 若需该段也“收敛得漂亮”，建议进一步减步长或采用更保结构的时间积分（辛方法）并控制数值阻尼。

- 选型建议
  - 若主要关注接触初期与随后的几次主振动：`Δt = 1e-3`（Verlet）或 `Δt = 5e-4`（RK4）通常足够。
  - 若需要长时间尾段的精细相位与能量：适度减小 `Δt` 或采用辛积分，并结合小的数值阻尼。

## 许可协议
本项目采用 MIT 许可，详见 `LICENSE`。
