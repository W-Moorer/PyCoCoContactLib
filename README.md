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
│   ├── ballMesh.obj              # 示例网格（球）
│   └── ballMesh/                 # 运行后生成的结果目录
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
python -m demos.demo_drop \
  --mesh models/ballMesh.obj \
  --time 8.0 \
  --dt 1e-2 \
  --sub 1e-3 \
  --out mesh_drop_results_rapid_verlet.csv \
  --log rapid_output_verlet.log \
  --progress 0.01 \
  --verbose
```

- 阻尼/半波阻尼（对应原始 damped 演示）：
```
python -m demos.demo_drop \
  --mesh models/ballMesh.obj \
  --time 8.0 \
  --dt 0.01 \
  --sub 5e-4 \
  --damp 100.0 \
  --half_wave \
  --out mesh_drop_results_rapid_verlet_damped.csv \
  --log rapid_output_verlet_damped.log \
  --progress 1.0 \
  --verbose
```

运行成功后，会在与 `ballMesh.obj` 同目录下新建 `models/ballMesh/` 子目录，内含 CSV、日志与绘图。替换网格时，将 `--mesh` 指向你的 OBJ 文件路径。

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

## 许可协议
本项目采用 MIT 许可，详见 `LICENSE`。
