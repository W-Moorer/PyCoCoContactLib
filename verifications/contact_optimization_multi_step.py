#!/usr/bin/env python3
"""
接触优化机制多步长对比测试
基于test_contact_optimization.py，将不同时间步长的结果绘制在一张图上进行对比分析
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import sys
import os

# 添加rcsim路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rcsim.physics.integrators.implicit_integrators.generalized_alpha import simulate
from rcsim.physics.world import ContactWorld as World
from rcsim.physics.rigid_bodies import RigidMeshBody
from rcsim.contact.mesh import RapidMesh

def create_test_scene():
    """创建测试场景：球体下落碰撞地面"""
    
    # 创建简单的网格对象（用于接触检测）
    # 地面网格：一个简单的平面三角形
    ground_vertices = [
        np.array([-10.0, -10.0, 0.0], dtype=float),
        np.array([10.0, -10.0, 0.0], dtype=float),
        np.array([10.0, 10.0, 0.0], dtype=float),
        np.array([-10.0, 10.0, 0.0], dtype=float)
    ]
    ground_triangles = [(0, 1, 2), (0, 2, 3)]
    ground_mesh = RapidMesh(ground_vertices, ground_triangles)
    
    # 球体网格：一个简单的四面体近似球体
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
    
    # 创建球体 - 将初始高度降低到0.8米，确保球体能够接触到地面
    ball = RigidMeshBody(mass=1.0, position=[0.0, 0.0, 0.8], velocity=[0.0, 0.0, 0.0], mesh=ball_mesh)  # 初始高度0.8米
    
    # 创建World对象并传入bodies列表
    world = World(bodies=[ground, ball])
    
    # 设置重力（使用标准重力加速度）
    world.g = -9.81  # 标准重力加速度
    
    # 建立接触检测对
    world.build_all_pairs()
    
    return world

def simulate_with_dt(dt_sub, t_end=1.0, use_performance_optimization=True):
    """使用指定子步长进行模拟"""
    world = create_test_scene()
    
    # 模拟参数 - 使用与test_contact_optimization.py相同的参数
    dt_frame = 0.01  # 固定帧步长
    
    # 运行模拟
    contact_history = []
    bounce_history = []
    
    def on_substep_callback(t, world):
        # 记录接触状态
        if len(world.bodies) >= 2:
            ball_z = world.bodies[1].body.position[2]
            ball_vz = world.bodies[1].body.velocity[2]
            
            contact_info = {
                't': t,
                'ball_z': ball_z,
                'ball_vz': ball_vz
            }
            contact_history.append(contact_info)
            
            # 检测反弹事件（速度从负变正）
            if len(contact_history) > 1:
                prev_vz = contact_history[-2]['ball_vz']
                if prev_vz < 0 and ball_vz > 0:
                    bounce_history.append({
                        't': t,
                        'ball_z': ball_z,
                        'ball_vz': ball_vz
                    })
    
    try:
        print(f"  开始模拟: dt_sub={dt_sub:.6f}s, dt_frame={dt_frame}s, t_end={t_end}s")
        simulate(
            world, t_end, dt_frame, dt_sub,
            on_substep=on_substep_callback,
            rho_inf=0.0,
            use_performance_optimization=use_performance_optimization
        )
        print(f"  模拟完成，步数: {len(contact_history)}")
        return contact_history, bounce_history, True
    except Exception as e:
        print(f"  模拟错误: {e}")
        import traceback
        traceback.print_exc()
        return contact_history, bounce_history, False

def analyze_simulation_result(contact_history, bounce_history, dt_sub):
    """分析单个模拟的结果"""
    if not contact_history:
        return None
    
    # 计算能量
    energies = []
    for info in contact_history:
        kinetic = 0.5 * 1.0 * info['ball_vz']**2  # 动能
        potential = 1.0 * 9.81 * info['ball_z']   # 势能
        energies.append(kinetic + potential)
    
    # 初始能量
    initial_energy = energies[0]
    final_energy = energies[-1]
    energy_error = abs(final_energy - initial_energy) / initial_energy * 100
    
    # 检测接触事件
    contact_events = []
    for i in range(1, len(contact_history)):
        prev_z = contact_history[i-1]['ball_z']
        curr_z = contact_history[i]['ball_z']
        
        # 接触条件：球体底部接触地面
        ball_bottom_z = curr_z - 0.5  # 网格最低点z=-0.5
        if ball_bottom_z <= 0 and prev_z > 0.5:
            contact_events.append({
                't': contact_history[i]['t'],
                'ball_z': curr_z,
                'ball_vz': contact_history[i]['ball_vz']
            })
    
    # 分析反弹行为
    bounce_heights = [b['ball_z'] for b in bounce_history]
    
    return {
        'dt_sub': dt_sub,
        'contact_history': contact_history,
        'bounce_history': bounce_history,
        'initial_energy': initial_energy,
        'final_energy': final_energy,
        'energy_error': energy_error,
        'contact_events': contact_events,
        'bounce_heights': bounce_heights,
        'num_steps': len(contact_history)
    }

def plot_multi_step_comparison(results):
    """绘制多步长对比图"""
    # 设置Times字体
    rcParams['font.family'] = 'Times New Roman'
    rcParams['font.size'] = 12
    rcParams['axes.unicode_minus'] = False
    
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 定义颜色和线型
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    line_styles = ['-', '--', '-.', ':', '-', '--']
    
    # 绘制位置对比图
    ax1 = axes[0, 0]
    for i, result in enumerate(results):
        times = [h['t'] for h in result['contact_history']]
        positions = [h['ball_z'] for h in result['contact_history']]
        label = f'dt={result["dt_sub"]:.4f}s'
        ax1.plot(times, positions, color=colors[i], linestyle=line_styles[i], 
                linewidth=2, label=label)
    
    ax1.axhline(y=0.5, color='black', linestyle='--', linewidth=1, label='Ground Level (z=0.5)')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Position (m)')
    ax1.set_title('Ball Position vs Time (Contact Optimization)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 绘制速度对比图
    ax2 = axes[0, 1]
    for i, result in enumerate(results):
        times = [h['t'] for h in result['contact_history']]
        velocities = [h['ball_vz'] for h in result['contact_history']]
        label = f'dt={result["dt_sub"]:.4f}s'
        ax2.plot(times, velocities, color=colors[i], linestyle=line_styles[i], 
                linewidth=2, label=label)
    
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, label='Zero Velocity')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Velocity (m/s)')
    ax2.set_title('Ball Velocity vs Time (Contact Optimization)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 绘制能量对比图
    ax3 = axes[1, 0]
    for i, result in enumerate(results):
        times = [h['t'] for h in result['contact_history']]
        energies = []
        for h in result['contact_history']:
            kinetic = 0.5 * 1.0 * h['ball_vz']**2
            potential = 1.0 * 9.81 * h['ball_z']
            energies.append(kinetic + potential)
        
        label = f'dt={result["dt_sub"]:.4f}s (Error: {result["energy_error"]:.2f}%)'
        ax3.plot(times, energies, color=colors[i], linestyle=line_styles[i], 
                linewidth=2, label=label)
    
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Energy (J)')
    ax3.set_title('Energy Conservation (Contact Optimization)', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # 绘制反弹高度对比图
    ax4 = axes[1, 1]
    for i, result in enumerate(results):
        if result['bounce_heights']:
            bounce_numbers = list(range(len(result['bounce_heights'])))
            label = f'dt={result["dt_sub"]:.4f}s ({len(result["bounce_heights"]):d} bounces)'
            ax4.plot(bounce_numbers, result['bounce_heights'], 
                    color=colors[i], linestyle=line_styles[i], 
                    marker='o', markersize=6, linewidth=2, label=label)
    
    ax4.set_xlabel('Bounce Number')
    ax4.set_ylabel('Bounce Height (m)')
    ax4.set_title('Bounce Height Decay (Contact Optimization)', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('contact_optimization_multi_step.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def plot_convergence_analysis(results):
    """绘制收敛性分析图"""
    # 设置Times字体
    rcParams['font.family'] = 'Times New Roman'
    rcParams['font.size'] = 12
    rcParams['axes.unicode_minus'] = False
    
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 提取数据
    dt_values = [r['dt_sub'] for r in results]
    energy_errors = [r['energy_error'] for r in results]
    num_steps = [r['num_steps'] for r in results]
    num_bounces = [len(r['bounce_heights']) for r in results]
    
    # 能量误差收敛性
    ax1 = axes[0, 0]
    ax1.loglog(dt_values, energy_errors, 'o-', linewidth=2, markersize=8)
    ax1.set_xlabel('Time Step (s)')
    ax1.set_ylabel('Energy Error (%)')
    ax1.set_title('Energy Error vs Time Step', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, which='both')
    
    # 添加收敛率分析
    for i, (dt, error) in enumerate(zip(dt_values, energy_errors)):
        ax1.annotate(f'{error:.2f}%', (dt, error), xytext=(5, 5), textcoords='offset points')
    
    # 计算步数对比
    ax2 = axes[0, 1]
    ax2.semilogx(dt_values, num_steps, 's-', linewidth=2, markersize=8)
    ax2.set_xlabel('Time Step (s)')
    ax2.set_ylabel('Number of Steps')
    ax2.set_title('Computational Cost vs Time Step', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 反弹次数对比
    ax3 = axes[1, 0]
    ax3.semilogx(dt_values, num_bounces, '^-', linewidth=2, markersize=8)
    ax3.set_xlabel('Time Step (s)')
    ax3.set_ylabel('Number of Bounces')
    ax3.set_title('Bounce Behavior vs Time Step', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 效率分析（误差/步数）
    ax4 = axes[1, 1]
    efficiency = [error/steps for error, steps in zip(energy_errors, num_steps)]
    ax4.semilogx(dt_values, efficiency, 'd-', linewidth=2, markersize=8)
    ax4.set_xlabel('Time Step (s)')
    ax4.set_ylabel('Energy Error per Step')
    ax4.set_title('Computational Efficiency Analysis', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('contact_optimization_convergence.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def main():
    """主函数：运行多步长对比测试"""
    print("=== Contact Optimization Multi-step Test ===")
    
    # 定义不同的时间步长
    dt_values = [0.001, 0.0005, 0.00025, 0.000125]
    
    results = []
    
    for dt_sub in dt_values:
        print(f"\nRunning simulation with dt_sub = {dt_sub:.6f} s")
        contact_history, bounce_history, success = simulate_with_dt(dt_sub)
        
        if success and contact_history:
            result = analyze_simulation_result(contact_history, bounce_history, dt_sub)
            if result:
                results.append(result)
                print(f"  Steps: {result['num_steps']}")
                print(f"  Energy error: {result['energy_error']:.2f}%")
                print(f"  Contact events: {len(result['contact_events'])}")
                print(f"  Bounce events: {len(result['bounce_heights'])}")
                if result['bounce_heights']:
                    print(f"  Bounce heights: {[f'{h:.3f}' for h in result['bounce_heights']]}")
        else:
            print(f"  Simulation failed")
    
    # 绘制对比图
    if results:
        print("\nGenerating multi-step comparison plots...")
        plot_multi_step_comparison(results)
        plot_convergence_analysis(results)
        
        # 输出汇总信息
        print("\n=== Summary ===")
        for result in results:
            print(f"dt={result['dt_sub']:.6f}s: "
                  f"Steps={result['num_steps']}, "
                  f"Energy Error={result['energy_error']:.2f}%, "
                  f"Contacts={len(result['contact_events'])}, "
                  f"Bounces={len(result['bounce_heights'])}")
        
        print("\nContact optimization multi-step comparison plots saved as:")
        print("  'contact_optimization_multi_step.png'")
        print("  'contact_optimization_convergence.png'")
    else:
        print("No successful simulations to plot")

if __name__ == "__main__":
    main()