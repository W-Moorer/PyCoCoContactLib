"""
优化的隐式积分器实现
针对变步长机制和接触检测频率进行性能优化
"""
import sys
import os
import time
import numpy as np
from typing import List, Tuple, Optional, Callable
import logging
from datetime import datetime
import matplotlib.pyplot as plt

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rcsim.contact import RapidMesh
from rcsim.physics import RigidMeshBody, ContactWorld
from rcsim.physics.integrators.implicit_integrators.generalized_alpha import (
    step_sub, ContactPerformanceOptimizer
)
from rcsim.contact.contact_types import ContactManifold, CollisionResult
from rcsim.contact.tri_geom import triTriIntersect

class OptimizedContactPerformanceOptimizer(ContactPerformanceOptimizer):
    """优化的性能优化器"""
    
    def __init__(self):
        super().__init__()
        # 添加智能步长预测参数
        self.smart_prediction_enabled = True
        self.prediction_history = []
        self.max_history_size = 50
        self.adaptive_threshold = 0.1  # 自适应阈值
        
    def smart_time_to_collide_prediction(self, world: ContactWorld, dt: float) -> float:
        """智能碰撞时间预测"""
        # 首先调用父类的预测方法
        base_prediction = super().smart_time_to_collide_prediction(world, dt)
        
        if not self.smart_prediction_enabled:
            return base_prediction
        
        # 基于历史数据进行智能预测
        if len(self.prediction_history) >= 2:
            # 使用线性回归预测
            times = np.array([h[0] for h in self.prediction_history[-10:]])
            predictions = np.array([h[1] for h in self.prediction_history[-10:]])
            
            if len(times) >= 2:
                # 计算趋势
                trend = np.polyfit(times, predictions, 1)
                predicted_time = np.polyval(trend, times[-1] + dt)
                
                # 应用自适应阈值
                if predicted_time < self.adaptive_threshold:
                    return max(predicted_time, 1e-6)
        
        # 回退到基础预测
        return base_prediction
    
    def update_prediction_history(self, timestamp: float, predicted_time: float):
        """更新预测历史"""
        self.prediction_history.append((timestamp, predicted_time))
        if len(self.prediction_history) > self.max_history_size:
            self.prediction_history.pop(0)
    
    def handle_step_rejection(self, current_dt: float, min_dt: float) -> float:
        """处理步长拒绝"""
        # 基于历史拒绝率自适应调整步长
        if len(self.prediction_history) >= 3:
            # 分析最近的拒绝模式
            recent_rejections = len([h for h in self.prediction_history[-5:] if h[1] < 1e-4])
            
            if recent_rejections >= 2:  # 连续拒绝
                return max(min_dt, current_dt * 0.3)
            else:
                return max(min_dt, current_dt * 0.7)
        else:
            return max(min_dt, current_dt * 0.5)

class AdaptiveContactDetector:
    """自适应接触检测器"""
    
    def __init__(self, world: ContactWorld):
        self.world = world
        self.last_full_detection_time = 0.0
        self.full_detection_interval = 0.01  # 完整检测间隔
        self.quick_check_threshold = 0.001  # 快速检查阈值
        
    def smart_contact_detection(self, current_time: float) -> bool:
        """智能接触检测"""
        # 检查是否需要完整检测
        if current_time - self.last_full_detection_time >= self.full_detection_interval:
            self.world.build_all_pairs()
            self.last_full_detection_time = current_time
            return True
        
        # 快速检查：只检查已知接触对
        contact_detected = False
        for pair in self.world.pairs:
            if pair.manifold and len(pair.manifold.points) > 0:
                # 检查接触点是否仍然有效
                poseA = (self.world.bodies[pair.i].body.position, self.world.bodies[pair.i].R)
                poseB = (self.world.bodies[pair.j].body.position, self.world.bodies[pair.j].R)
                
                # 快速距离检查
                if self.quick_distance_check(pair, poseA, poseB):
                    contact_detected = True
                    break
        
        return contact_detected
    
    def accurate_contact_detection(self, world):
        """精确接触检测：基于实际穿透深度"""
        # 使用measure_penetration的检测逻辑
        current_pen = measure_penetration(world)
        penetration_threshold = 1.0e-4 * 5  # 0.0005m
        return current_pen > penetration_threshold
    
    def quick_distance_check(self, pair, poseA, poseB) -> bool:
        """快速距离检查 - 修复：与measure_penetration保持一致"""
        try:
            # 关键修复：直接检查接触流形的穿透深度，与measure_penetration逻辑一致
            if pair.manifold and pair.manifold.points:
                for cp in pair.manifold.points:
                    if cp.phi < 0.0:  # 穿透深度为负表示接触
                        return True  # 只要有一个接触点有穿透，就认为有接触
            
            # 如果流形信息不可用，使用边界球作为后备
            centerA = poseA[0]
            centerB = poseB[0]
            distance = np.linalg.norm(np.array(centerA) - np.array(centerB))
            
            # 使用更宽松的阈值，确保能检测到接触
            return distance < self.quick_check_threshold * 2.0  # 放宽阈值
        except:
            return False

# 设置日志记录器
def setup_contact_logger():
    """设置接触检测日志记录器"""
    logger = logging.getLogger('contact_detection')
    logger.setLevel(logging.DEBUG)
    
    # 如果已经有处理器，则清除
    if logger.handlers:
        logger.handlers.clear()
    
    # 创建文件处理器
    log_filename = f"contact_detection_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    
    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.propagate = False  # 防止传播到根日志记录器
    
    return logger, log_filename

# 全局日志记录器
contact_logger, log_file_path = setup_contact_logger()

# 在measure_penetration函数中添加调试
def measure_penetration(world: ContactWorld) -> float:
    pen = 0.0
    contact_pairs_info = []
    
    for pair_idx, pair in enumerate(world.pairs):
        pair_pen = 0.0
        contact_points_info = []
        
        for cp_idx, cp in enumerate(pair.manifold.points):
            if cp.phi < 0.0:
                current_pen = -cp.phi
                pen = max(pen, current_pen)
                pair_pen = max(pair_pen, current_pen)
                contact_points_info.append({
                    'point_index': cp_idx,
                    'phi': cp.phi,
                    'penetration': current_pen,
                    'position': cp.position.tolist() if hasattr(cp.position, 'tolist') else list(cp.position)
                })
        
        if pair_pen > 0:
            contact_pairs_info.append({
                'pair_index': pair_idx,
                'body_i': pair.i,
                'body_j': pair.j,
                'max_penetration': pair_pen,
                'contact_points': contact_points_info
            })
    
    # 记录详细的接触信息
    if contact_pairs_info:
        contact_logger.info(f"接触检测结果 - 最大穿透深度: {pen:.6e}")
        for pair_info in contact_pairs_info:
            contact_logger.debug(f"  接触对{pair_info['pair_index']} (体{pair_info['body_i']}-体{pair_info['body_j']}): 最大穿透={pair_info['max_penetration']:.6e}")
            for cp_info in pair_info['contact_points']:
                contact_logger.debug(f"    接触点{cp_info['point_index']}: phi={cp_info['phi']:.6e}, 穿透={cp_info['penetration']:.6e}")
    
    return pen

def measure_penetration_safe(world: ContactWorld) -> tuple[bool, float]:
    pen = 0.0
    near_band = getattr(world, "_step_max_move", None)
    contact_pairs_info = []
    
    # 第一阶段：基于流形的接触检测
    for pair_idx, pair in enumerate(world.pairs):
        i, j = pair.i, pair.j
        A = world.bodies[i]
        B = world.bodies[j]
        poseA = (A.body.position, A.R)
        poseB = (B.body.position, B.R)
        
        had_contact_prev = False
        if pair.manifold and pair.manifold.points:
            for _cp in pair.manifold.points:
                if _cp.phi < 0.0:
                    had_contact_prev = True
                    break
        if near_band is not None:
            if had_contact_prev:
                try:
                    pair.detector._near_band = 0.0
                except Exception:
                    pair.detector._near_band = 0.0
            else:
                try:
                    pair.detector._near_band = 0.5 * float(near_band)
                except Exception:
                    pair.detector._near_band = float(near_band)
        
        # 使用新的query_manifold方法
        manifold = pair.detector.query_manifold(poseA, poseB)
        pair.manifold = manifold
        
        pair_pen = 0.0
        contact_points_info = []
        
        for cp_idx, cp in enumerate(manifold.points):
            if cp.phi < 0.0:
                current_pen = -cp.phi
                pen = max(pen, current_pen)
                pair_pen = max(pair_pen, current_pen)
                contact_points_info.append({
                    'point_index': cp_idx,
                    'phi': cp.phi,
                    'penetration': current_pen,
                    'position': cp.pB_world.tolist() if hasattr(cp.pB_world, 'tolist') else list(cp.pB_world)
                })
        
        if pair_pen > 0:
            contact_pairs_info.append({
                'pair_index': pair_idx,
                'body_i': i,
                'body_j': j,
                'detection_method': 'manifold',
                'max_penetration': pair_pen,
                'contact_points': contact_points_info
            })
    
    if pen > 0.0:
        contact_logger.info(f"第一阶段(流形)接触检测 - 最大穿透深度: {pen:.6e}")
        for pair_info in contact_pairs_info:
            contact_logger.debug(f"  接触对{pair_info['pair_index']} (体{pair_info['body_i']}-体{pair_info['body_j']}): 最大穿透={pair_info['max_penetration']:.6e}")
            for cp_info in pair_info['contact_points']:
                contact_logger.debug(f"    接触点{cp_info['point_index']}: phi={cp_info['phi']:.6e}, 穿透={cp_info['penetration']:.6e}")
        return True, pen
    
    # 第二阶段：基于三角形相交的接触检测
    pen2 = 0.0
    triangle_contact_pairs_info = []
    
    for pair_idx, pair in enumerate(world.pairs):
        i, j = pair.i, pair.j
        A = world.bodies[i]
        B = world.bodies[j]
        poseA = (A.body.position, A.R)
        poseB = (B.body.position, B.R)
        
        pairs_idx: list[tuple[int, int]] = []
        pair.detector.meshA.bvh.find_collision(pair.detector.meshB.bvh, poseA, poseB, pairs_idx)
        
        if not pairs_idx:
            continue
        
        pair_triangle_pen = 0.0
        triangle_contacts_info = []
        
        for triangle_idx, (ia, ib) in enumerate(pairs_idx[:1000]):
            nodesA = A.mesh.get_triangle_nodes(ia, poseA)
            nodesB = B.mesh.get_triangle_nodes(ib, poseB)
            res = CollisionResult()
            triTriIntersect(nodesA, nodesB, res)
            
            if res.contPtsPairs:
                n = res.contactNormal.copy()
                nn = float(np.linalg.norm(n))
                if nn < 1e-18:
                    continue
                n = n / nn
                
                triangle_pen = 0.0
                contact_points_info = []
                
                # 1) 基于接触法向的点对投影深度
                for pt_idx, (pA_world, pB_world) in enumerate(res.contPtsPairs):
                    v = pB_world - pA_world
                    depth = float(np.dot(v, n))
                    if depth > triangle_pen:
                        triangle_pen = depth
                    contact_points_info.append({
                        'point_pair_index': pt_idx,
                        'depth': depth,
                        'pointA': pA_world.tolist() if hasattr(pA_world, 'tolist') else list(pA_world),
                        'pointB': pB_world.tolist() if hasattr(pB_world, 'tolist') else list(pB_world)
                    })
                
                # 2) 节点到接触平面最大穿透
                mid_sum = np.zeros(3, dtype=float)
                cnt = 0
                for pA_world, pB_world in res.contPtsPairs:
                    mid_sum += 0.5 * (pA_world + pB_world)
                    cnt += 1
                p0 = (mid_sum / float(cnt)) if cnt > 0 else sum((p for p, _ in res.contPtsPairs), np.zeros(3, dtype=float)) / max(1, len(res.contPtsPairs))
                
                # A节点穿透检测
                for k in range(3):
                    sA = float(np.dot(nodesA[k] - p0, n))
                    if sA < 0.0:
                        triangle_pen = max(triangle_pen, -sA)
                
                # B节点穿透检测
                for k in range(3):
                    sB = float(np.dot(nodesB[k] - p0, n))
                    if sB > 0.0:
                        triangle_pen = max(triangle_pen, sB)
                
                if triangle_pen > pair_triangle_pen:
                    pair_triangle_pen = triangle_pen
                
                triangle_contacts_info.append({
                    'triangle_pair': (ia, ib),
                    'penetration': triangle_pen,
                    'contact_points': contact_points_info
                })
        
        if pair_triangle_pen > 0:
            pen2 = max(pen2, pair_triangle_pen)
            triangle_contact_pairs_info.append({
                'pair_index': pair_idx,
                'body_i': i,
                'body_j': j,
                'detection_method': 'triangle_intersection',
                'max_penetration': pair_triangle_pen,
                'triangle_contacts': triangle_contacts_info
            })
    
    if pen2 > 0.0:
        # 结合近域带宽对本步几何变化进行上限夹持
        try:
            lim = float(near_band) if near_band is not None else None
        except Exception:
            lim = None
        
        original_pen2 = pen2
        if lim is not None:
            pen2 = min(pen2, lim + 1e-6)
        
        contact_logger.info(f"第二阶段(三角形相交)接触检测 - 原始最大穿透: {original_pen2:.6e}, 限制后: {pen2:.6e}")
        for pair_info in triangle_contact_pairs_info:
            contact_logger.debug(f"  接触对{pair_info['pair_index']} (体{pair_info['body_i']}-体{pair_info['body_j']}): 最大穿透={pair_info['max_penetration']:.6e}")
            for tri_info in pair_info['triangle_contacts']:
                contact_logger.debug(f"    三角形对{tri_info['triangle_pair']}: 穿透={tri_info['penetration']:.6e}")
        
        return True, pen2
    
    contact_logger.debug("无接触检测")
    return False, 0.0

def optimized_simulate(
    world: ContactWorld,
    t_end: float,
    dt_frame: float = 0.01,
    dt_sub: float = 0.001,
    on_substep: Optional[Callable] = None,
    min_dt: float = 1.0e-6,
    # 新增参数：从分离态到接触态的转换阈值
    separation_to_contact_threshold: float = 1e-3,
    # 新增参数：接触态维持时的阈值
    contact_maintain_threshold: float = 1e-2,
    # 新增参数：接触态转换后的步长松弛系数
    contact_step_relax_factor: float = 2.0,
    rho_inf: float = 0.5,  # 关键修复：添加rho_inf参数，与原始积分器保持一致
    max_iter: int = 20,     # 关键修复：添加max_iter参数
    tol: float = 1e-8       # 关键修复：添加容差参数
) -> None:
    """优化的隐式积分器模拟函数"""
    
    performance_stats = {
        'total_steps': 0,
        'rejected_steps': 0,
        'step_times': []
    }
    
    t = 0.0
    dt_current = float(dt_sub)
    prev_contact = False
    prev_pen = 0.0
    com_times = []
    com_z = []
    def _primary_body():
        for rb in world.bodies:
            if not rb.body.is_static:
                return rb.body
        return world.bodies[-1].body
    def _primary_index():
        for i, rb in enumerate(world.bodies):
            if not rb.body.is_static:
                return i
        return len(world.bodies) - 1
    def _print_accept(t_cur, dt_used, prev_ct, post_ct, prev_p, post_p, dt_next):
        b = _primary_body()
        dpen = abs(post_p - prev_p)
        F = world.compute_forces()
        Fg = np.zeros_like(F)
        for i, rb in enumerate(world.bodies):
            if not rb.body.is_static:
                Fg[i, 2] = rb.body.mass * world.g
        Fc = F - Fg
        pid = _primary_index()
        fmag = float(np.linalg.norm(Fc[pid]))
        print(f"[接受] t={t_cur:.6f}s dt={dt_used:.6e} 上接触={int(prev_ct)} 本接触={int(post_ct)} prev_pen={prev_p:.6e} post_pen={post_p:.6e} dpen={dpen:.6e} 下步dt={dt_next:.6e} z={b.position[2]:.6f} vz={b.velocity[2]:.6f} Fmag={fmag:.6e}")
    def _print_reject(t_cur, dt_used, reason, prev_ct, post_ct, prev_p, post_p, dt_next):
        F = world.compute_forces()
        Fg = np.zeros_like(F)
        for i, rb in enumerate(world.bodies):
            if not rb.body.is_static:
                Fg[i, 2] = rb.body.mass * world.g
        Fc = F - Fg
        pid = _primary_index()
        fmag = float(np.linalg.norm(Fc[pid]))
        print(f"[拒绝] t={t_cur:.6f}s dt={dt_used:.6e} 原因={reason} 上接触={int(prev_ct)} 本接触={int(post_ct)} prev_pen={prev_p:.6e} post_pen={post_p:.6e} 新dt={dt_next:.6e} Fmag={fmag:.6e}")
    def _print_min_dt_warning(t_cur, dt_used):
        print(f"[最小步长警告] t={t_cur:.6f}s dt={dt_used:.6e} 已达到下限并直接采用")

    def _capture_state():
        X = np.stack([rb.body.position.copy() for rb in world.bodies], axis=0)
        V = np.stack([rb.body.velocity.copy() for rb in world.bodies], axis=0)
        return X, V

    def _restore_state(X, V):
        # 回滚几何状态
        for i, rb in enumerate(world.bodies):
            b = rb.body
            if b.is_static:
                continue
            b.position = X[i].copy()
            b.velocity = V[i].copy()
        
        # 关键修复：重置所有接触对的流形
        for pair in world.pairs:
            pair.manifold = ContactManifold()  # 清空接触流形
    
    # 记录模拟开始信息
    contact_logger.info(f"=== 模拟开始 ===")
    contact_logger.info(f"目标时间: {t_end:.6f}s, 帧步长: {dt_frame:.6e}s, 子步长: {dt_sub:.6e}s")
    contact_logger.info(f"最小步长: {min_dt:.6e}s, 接触阈值: {separation_to_contact_threshold:.6e}")
    contact_logger.info(f"日志文件: {log_file_path}")
    
    # 主模拟循环
    max_total_steps = int(t_end / min_dt * 10)  # 添加最大步数保护
    step_count = 0
    
    while t < t_end and step_count < max_total_steps:
        step_count += 1
        
        # 记录当前步信息
        contact_logger.info(f"--- 步进 {step_count} ---")
        contact_logger.info(f"当前时间: {t:.6f}s, 剩余时间: {t_end - t:.6e}s")
        
        # 检查是否接近结束时间 - 放宽条件确保能达到最终时间
        if t_end - t < 1e-8:
            # 如果剩余时间很小，直接完成当前帧
            remaining = t_end - t
            if remaining > 0:
                # 使用剩余时间完成最后一步
                dt_try = remaining
                contact_logger.info(f"执行最后一步: dt={dt_try:.6e}s")
                X0, V0 = _capture_state()
                vmaxi = float(np.max([float(np.linalg.norm(rb.body.velocity)) for rb in world.bodies if not rb.body.is_static] + [0.0]))
                world._step_max_move = dt_try * vmaxi + 0.5 * (dt_try ** 2) * abs(world.g)
                world._force_fresh_manifold = True
                sub_start = time.time()
                step_sub(world, dt_try, rho_inf=rho_inf, max_iter=max_iter, tol=tol)
                t += dt_try
                performance_stats['total_steps'] += 1
                performance_stats['step_times'].append(time.time() - sub_start)
                if on_substep is not None:
                    on_substep(t, world)
                contact_logger.info(f"[最后一步] t={t:.6f}s dt={dt_try:.6e} 完成最终步长")
                print(f"[最后一步] t={t:.6f}s dt={dt_try:.6e} 完成最终步长")
            break
            
        remaining = min(dt_frame, t_end - t)
        
        # 内层循环：处理当前帧内的子步
        while remaining > 1e-10:  # 提高精度阈值
            # 防止步长过小导致无限循环
            if dt_current < min_dt * 1.1:
                dt_current = min_dt
                
            dt_try = min(dt_current, remaining)
            
            # 如果尝试步长过小，直接使用剩余时间
            if dt_try < min_dt * 0.5:
                dt_try = remaining
            
            # 记录子步信息
            contact_logger.info(f"子步尝试: dt={dt_try:.6e}s, 当前步长={dt_current:.6e}s, 剩余帧时间={remaining:.6e}s")
            contact_logger.info(f"前状态: 接触={prev_contact}, 穿透深度={prev_pen:.6e}")
                
            X0, V0 = _capture_state()
            vmaxi = float(np.max([float(np.linalg.norm(rb.body.velocity)) for rb in world.bodies if not rb.body.is_static] + [0.0]))
            world._step_max_move = dt_try * vmaxi + 0.5 * (dt_try ** 2) * abs(world.g)
            world._force_fresh_manifold = True
            sub_start = time.time()
            step_sub(world, dt_try, rho_inf=rho_inf, max_iter=max_iter, tol=tol)
            in_contact_safe, post_pen = measure_penetration_safe(world)
            post_contact = in_contact_safe
            Vnew = np.stack([rb.body.velocity.copy() for rb in world.bodies], axis=0)
            vel_change = float(np.max([float(np.linalg.norm(Vnew[i] - V0[i])) for i, rb in enumerate(world.bodies) if not rb.body.is_static] + [0.0]))
            abn_thr = max(1.0, 10.0 * abs(world.g) * dt_try)
            
            # 记录步进后状态
            contact_logger.info(f"步进后状态: 接触={post_contact}, 穿透深度={post_pen:.6e}, 速度变化={vel_change:.6e}")

            if (not prev_contact) and (not post_contact) and (vel_change > abn_thr):
                contact_logger.warning(f"速度异常检测: 速度变化{vel_change:.6e} > 阈值{abn_thr:.6e}")
                _restore_state(X0, V0)
                new_dt = max(float(min_dt), 0.5 * dt_try)
                if new_dt <= float(min_dt) + 1e-12:
                    contact_logger.warning("达到最小步长限制")
                    _print_min_dt_warning(t, new_dt)
                    vmaxi2 = float(np.max([float(np.linalg.norm(rb.body.velocity)) for rb in world.bodies if not rb.body.is_static] + [0.0]))
                    world._step_max_move = new_dt * vmaxi2 + 0.5 * (new_dt ** 2) * abs(world.g)
                    world._force_fresh_manifold = True
                    sub2_start = time.time()
                    step_sub(world, new_dt, rho_inf=rho_inf, max_iter=max_iter, tol=tol)
                    post_contact2, post_pen2 = measure_penetration_safe(world)
                    t += new_dt
                    remaining -= new_dt
                    performance_stats['total_steps'] += 1
                    performance_stats['step_times'].append(time.time() - sub2_start)
                    if on_substep is not None:
                        on_substep(t, world)
                    prev_contact = post_contact2
                    contact_logger.info(f"最小步长接受: 新接触={post_contact2}, 新穿透={post_pen2:.6e}")
                    _print_accept(t, new_dt, False, post_contact2, 0.0, post_pen2, float(min_dt))
                    prev_pen = post_pen2 if post_contact2 else 0.0
                    dt_current = float(min_dt)
                    continue
                dt_current = new_dt
                contact_logger.info(f"步长拒绝: 速度异常, 新步长={dt_current:.6e}s")
                _print_reject(t, dt_try, "速度异常", False, False, prev_pen, post_pen, dt_current)
                performance_stats['rejected_steps'] += 1
            accept = False
            
            if prev_contact and post_contact:
                embedding_change = abs(post_pen - prev_pen)
                vmaxi = float(np.max([float(np.linalg.norm(rb.body.velocity)) for rb in world.bodies if not rb.body.is_static] + [0.0]))
                growth_lim = max(float(contact_maintain_threshold), 0.1 * vmaxi * dt_try)
                prev_pen_old = prev_pen
                
                # 记录接触维持状态信息
                contact_logger.info(f"接触维持状态: 穿透变化={embedding_change:.6e}, 阈值={growth_lim:.6e}")
                
                if embedding_change <= growth_lim:
                    accept = True
                    # 优化步长增长：避免无限增长，设置上限
                    new_dt = dt_current * float(contact_step_relax_factor)
                    dt_current = min(float(dt_sub), new_dt)
                    prev_contact = True
                    prev_pen = post_pen
                    contact_logger.info(f"接触维持接受: 步长调整={new_dt:.6e}s, 新步长={dt_current:.6e}s")
                    _print_accept(t, dt_try, True, True, prev_pen_old, post_pen, dt_current)
                else:
                    _restore_state(X0, V0)
                    new_dt = max(float(min_dt), 0.5 * dt_try)
                    if new_dt <= float(min_dt) + 1e-12:
                        _print_min_dt_warning(t, new_dt)
                        vmaxi2 = float(np.max([float(np.linalg.norm(rb.body.velocity)) for rb in world.bodies if not rb.body.is_static] + [0.0]))
                        world._step_max_move = new_dt * vmaxi2 + 0.5 * (new_dt ** 2) * abs(world.g)
                        world._force_fresh_manifold = True
                        sub2_start = time.time()
                        step_sub(world, new_dt, rho_inf=rho_inf, max_iter=max_iter, tol=tol)
                        post_contact2, post_pen2 = measure_penetration_safe(world)
                        t += new_dt
                        remaining -= new_dt
                        performance_stats['total_steps'] += 1
                        performance_stats['step_times'].append(time.time() - sub2_start)
                        if on_substep is not None:
                            on_substep(t, world)
                        prev_contact = post_contact2
                        contact_logger.info(f"接触维持最小步长接受: 新接触={post_contact2}, 新穿透={post_pen2:.6e}")
                        _print_accept(t, new_dt, True, post_contact2, prev_pen_old, post_pen2, float(min_dt))
                        prev_pen = post_pen2 if post_contact2 else 0.0
                        dt_current = float(min_dt)
                        continue
                    dt_current = new_dt
                    contact_logger.warning(f"接触维持拒绝: 穿透变化过大, 新步长={dt_current:.6e}s")
                    _print_reject(t, dt_try, "维持阶段穿透变化过大", True, True, prev_pen, post_pen, dt_current)
                    performance_stats['rejected_steps'] += 1
            elif (not prev_contact) and post_contact:
                prev_pen_old = prev_pen
                
                # 记录分离到接触转换状态信息
                contact_logger.info(f"分离到接触转换: 当前穿透={post_pen:.6e}, 阈值={separation_to_contact_threshold:.6e}")
                
                if post_pen <= float(separation_to_contact_threshold):
                    accept = True
                    prev_contact = True
                    prev_pen = post_pen
                    # 接触状态下保持适当步长
                    dt_current = min(float(dt_sub), dt_current)
                    contact_logger.info(f"分离到接触转换接受: 新接触状态={prev_contact}, 新步长={dt_current:.6e}s")
                    _print_accept(t, dt_try, False, True, prev_pen_old, post_pen, dt_current)
                else:
                    _restore_state(X0, V0)
                    new_dt = max(float(min_dt), 0.5 * dt_try)
                    if new_dt <= float(min_dt) + 1e-12:
                        _print_min_dt_warning(t, new_dt)
                        vmaxi2 = float(np.max([float(np.linalg.norm(rb.body.velocity)) for rb in world.bodies if not rb.body.is_static] + [0.0]))
                        world._step_max_move = new_dt * vmaxi2 + 0.5 * (new_dt ** 2) * abs(world.g)
                        world._force_fresh_manifold = True
                        sub2_start = time.time()
                        step_sub(world, new_dt, rho_inf=rho_inf, max_iter=max_iter, tol=tol)
                        post_contact2, post_pen2 = measure_penetration_safe(world)
                        t += new_dt
                        remaining -= new_dt
                        performance_stats['total_steps'] += 1
                        performance_stats['step_times'].append(time.time() - sub2_start)
                        if on_substep is not None:
                            on_substep(t, world)
                        prev_contact = post_contact2
                        contact_logger.info(f"分离到接触转换最小步长接受: 新接触={post_contact2}, 新穿透={post_pen2:.6e}")
                        _print_accept(t, new_dt, False, post_contact2, prev_pen_old, post_pen2, float(min_dt))
                        prev_pen = post_pen2 if post_contact2 else 0.0
                        dt_current = float(min_dt)
                        continue
                    dt_current = new_dt
                    contact_logger.warning(f"分离到接触转换拒绝: 初始穿透过大, 新步长={dt_current:.6e}s")
                    _print_reject(t, dt_try, "初始接触穿透过大", False, True, prev_pen, post_pen, dt_current)
                    performance_stats['rejected_steps'] += 1
            elif prev_contact and (not post_contact):
                prev_pen_old = prev_pen
                accept = True
                prev_contact = False
                prev_pen = 0.0
                # 分离状态下适当增加步长，但避免无限增长
                new_dt = dt_current * float(contact_step_relax_factor)
                dt_current = min(float(dt_sub), new_dt)
                contact_logger.info(f"接触分离状态: 从接触状态分离, 新步长={dt_current:.6e}s")
                _print_accept(t, dt_try, True, False, prev_pen_old, post_pen, dt_current)
            else:
                prev_pen_old = prev_pen
                accept = True
                prev_contact = False
                prev_pen = 0.0
                # 无接触状态下适当增加步长
                new_dt = dt_current * float(contact_step_relax_factor)
                dt_current = min(float(dt_sub), new_dt)
                contact_logger.info(f"无接触状态: 保持分离状态, 新步长={dt_current:.6e}s")
                _print_accept(t, dt_try, False, False, prev_pen_old, post_pen, dt_current)
            
            if accept:
                t += dt_try
                remaining -= dt_try
                performance_stats['total_steps'] += 1
                performance_stats['step_times'].append(time.time() - sub_start)
                if on_substep is not None:
                    on_substep(t, world)
                bb = _primary_body()
                com_times.append(t)
                com_z.append(float(bb.position[2]))
        
        # 添加进度输出
        if t % 0.1 < dt_frame:
            print(f"进度: {t:.3f}/{t_end:.3f}s ({t/t_end*100:.1f}%)")
    
    # 循环结束检查
    if step_count >= max_total_steps:
        print(f"警告: 达到最大步数限制 {max_total_steps}，仿真提前结束")
    else:
        print(f"仿真正常结束，总时间: {t:.6f}s")
    if com_times:
        fig = plt.figure(figsize=(6,4))
        ax = fig.add_subplot(111)
        ax.plot(np.array(com_times), np.array(com_z), color='blue')
        ax.set_xlabel('t [s]')
        ax.set_ylabel('z [m]')
        base = os.path.splitext(os.path.basename(log_file_path))[0]
        fname = f"{base}_com_z.png"
        plt.tight_layout()
        plt.savefig(fname)
        plt.close(fig)
    
    

def test_optimized_integrator():
    """测试优化的隐式积分器"""
    print("开始测试优化的隐式积分器...")
    
    # 创建测试场景
    meshA = RapidMesh.from_obj("models/ballMesh.obj")
    meshB = RapidMesh.from_obj("models/ballMesh.obj")
    
    A = RigidMeshBody(meshA, 1.0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], is_static=True)
    B = RigidMeshBody(meshB, 1.0, [0.0, 0.0, 205.0], [0.0, 0.0, 0.0], is_static=False)
    
    world = ContactWorld(
        [A, B],
        g=-9.8,
        k_contact=1e5,
        c_damp=0.0,
        half_wave_damp=False,
        damp_type="linear",
        exponent=1.0,
        rebound_factor=0.0,
    )
    world.build_all_pairs()
    
    # 性能监控回调
    def performance_monitor(t, w):
        if t % 0.1 < 0.001:  # 每0.1秒输出一次
            print(f"时间: {t:.3f}s")
    
    # 运行优化后的模拟
    start_time = time.time()
    
    optimized_simulate(
        world,
        t_end=8.0,  # 确保至少2秒模拟时间
        dt_frame=0.01,
        dt_sub=0.001,
        on_substep=performance_monitor,
        min_dt=1.0e-6,
        rho_inf=0.0,
        # 新增参数：从分离态到接触态的转换阈值
        separation_to_contact_threshold=1e-3,
        # 新增参数：接触态维持时的阈值
        contact_maintain_threshold=1e-2,
        # 新增参数：接触态转换后的步长松弛系数
        contact_step_relax_factor=2.0
    )
    
    total_time = time.time() - start_time
    print(f"总模拟时间: {total_time:.3f} 秒")
    
    return total_time

if __name__ == "__main__":
    test_optimized_integrator()
