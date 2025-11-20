#!/usr/bin/env python3
"""
简单的接触测试，验证球体下落和接触检测
"""

import numpy as np
from rcsim.physics.rigid_bodies import RigidMeshBody
from rcsim.physics.world import ContactWorld as World
from rcsim.contact.mesh import RapidMesh
from demos.optimized_generalized_alpha import optimized_simulate

def simple_test():
    """简单的接触测试"""
    print("=== Simple Contact Test ===")
    
    # 创建地面网格
    ground_vertices = [
        np.array([10.0, -10.0, 0.0], dtype=float),
        np.array([10.0, 10.0, 0.0], dtype=float),
        np.array([-10.0, -10.0, 0.0], dtype=float),
        np.array([-10.0, 10.0, 0.0], dtype=float)
    ]
    ground_triangles = [(0, 1, 2), (0, 2, 3)]
    ground_mesh = RapidMesh(ground_vertices, ground_triangles)
    
    # 创建球体网格
    ball_vertices = [
        np.array([0.0, 0.0, 0.5], dtype=float),
        np.array([0.5, 0.0, -0.5], dtype=float),
        np.array([-0.5, 0.0, -0.5], dtype=float),
        np.array([0.0, 0.5, -0.5], dtype=float),
        np.array([0.0, -0.5, -0.5], dtype=float)
    ]
    ball_triangles = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (0, 1, 4), (0, 2, 4), (1, 2, 3), (1, 2, 4)]
    ball_mesh = RapidMesh(ball_vertices, ball_triangles)
    
    # 创建地面
    ground = RigidMeshBody(mass=float("inf"), position=[0.0, 0.0, 0.0], velocity=[0.0, 0.0, 0.0], is_static=True, mesh=ground_mesh)
    
    # 创建球体 - 初始高度1.0米
    ball = RigidMeshBody(mass=1.0, position=[0.0, 0.0, 1.0], velocity=[0.0, 0.0, 0.0], mesh=ball_mesh)
    
    # 创建World对象
    world = World(bodies=[ground, ball])
    world.g = -9.81  # 标准重力
    world.build_all_pairs()
    
    print(f"Initial state: Ball position z={ball.body.position[2]:.3f}m")
    
    # 模拟参数
    dt_frame = 0.01
    dt_sub = 0.001
    t_end = 1.0
    
    # 运行模拟
    history = []
    
    def on_substep(t, world):
        ball_z = ball.body.position[2]
        ball_vz = ball.body.velocity[2]
        history.append({'t': t, 'z': ball_z, 'vz': ball_vz})
        if len(history) % 50 == 0:
            print(f"t={t:.3f}s, z={ball_z:.3f}m, vz={ball_vz:.3f}m/s")
    
    try:
        print("Starting simulation...")
        optimized_simulate(world, t_end, dt_frame, dt_sub, on_substep=on_substep, rho_inf=0.0)
        print("Simulation completed")
        
        # 分析结果
        print(f"Total steps: {len(history)}")
        print(f"Final state: z={ball.body.position[2]:.3f}m, vz={ball.body.velocity[2]:.3f}m/s")
        
        # 检查是否接触地面
        ball_bottom_z = ball.body.position[2] - 0.5  # 球体底部位置
        if ball_bottom_z <= 0:
            print("✅ Ball contacted ground")
        else:
            print("❌ Ball did not contact ground")
        
        # 绘制收敛性分析图
        plot_convergence_analysis(history)
            
        return True
        
    except Exception as e:
        print(f"Simulation error: {e}")
        import traceback
        traceback.print_exc()
        return False

def plot_convergence_analysis(history):
    """绘制收敛性分析图"""
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    
    # 设置Times字体
    rcParams['font.family'] = 'Times New Roman'
    rcParams['font.size'] = 12
    rcParams['axes.unicode_minus'] = False
    
    # 提取数据
    times = [h['t'] for h in history]
    positions = [h['z'] for h in history]
    velocities = [h['vz'] for h in history]
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # 绘制位置图
    ax1.plot(times, positions, 'b-', linewidth=2, label='Ball Position (z)')
    ax1.axhline(y=0.5, color='r', linestyle='--', linewidth=1, label='Ground Level (z=0.5)')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Position (m)')
    ax1.set_title('Ball Position vs Time', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 绘制速度图
    ax2.plot(times, velocities, 'g-', linewidth=2, label='Ball Velocity (vz)')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Velocity (m/s)')
    ax2.set_title('Ball Velocity vs Time', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('simple_contact_convergence.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 分析收敛性
    print("\n=== Convergence Analysis ===")
    
    # 计算能量
    energies = []
    for h in history:
        kinetic = 0.5 * 1.0 * h['vz']**2  # 动能
        potential = 1.0 * 9.81 * h['z']   # 势能
        energies.append(kinetic + potential)
    
    # 初始能量
    initial_energy = energies[0]
    final_energy = energies[-1]
    energy_error = abs(final_energy - initial_energy) / initial_energy * 100
    
    print(f"Initial energy: {initial_energy:.4f} J")
    print(f"Final energy: {final_energy:.4f} J")
    print(f"Energy conservation error: {energy_error:.2f}%")
    
    # 检测接触时间
    contact_time = None
    for i, h in enumerate(history):
        if h['z'] <= 0.5:  # 球体底部接触地面
            contact_time = h['t']
            print(f"First contact time: {contact_time:.4f} s")
            break
    
    # 绘制能量图
    fig, ax3 = plt.subplots(figsize=(10, 6))
    ax3.plot(times, energies, 'purple', linewidth=2, label='Total Energy')
    ax3.axhline(y=initial_energy, color='orange', linestyle='--', linewidth=1, label='Initial Energy')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Energy (J)')
    ax3.set_title('Energy Conservation Analysis', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig('simple_contact_energy.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Convergence plots saved as 'simple_contact_convergence.png' and 'simple_contact_energy.png'")

if __name__ == "__main__":
    simple_test()
