import numpy as np
from typing import Callable, Optional

from rcsim.physics.world import ContactWorld
from rcsim.io.perf import perf
from rcsim.contact.aabb_bvh import Aabb
from rcsim.contact.tri_geom import triangle_distance, triTriIntersect
from rcsim.contact.contact_types import CollisionResult


def _params(rho_inf: float) -> tuple[float, float, float, float]:
    alpha_m = (1.0 - rho_inf) / (1.0 + rho_inf)
    alpha_f = 1.0 / (1.0 + rho_inf)
    gamma = 0.5 + alpha_f - alpha_m
    beta = 0.25 * (1.0 + alpha_f - alpha_m) ** 2
    return alpha_m, alpha_f, gamma, beta


def step_sub(
    world: ContactWorld,
    dt: float,
    *,
    rho_inf: float = 0.5,
    max_iter: int = 20,
    tol: float = 1e-8,
    relax: float = 1.0,
    use_newton: bool = False,
    diff_eps_x: float = 1e-6,
    diff_eps_v: float = 1e-6,
) -> None:
    n = len(world.bodies)
    n3 = n * 3
    x0 = np.stack([rb.body.position.copy() for rb in world.bodies], axis=0)
    v0 = np.stack([rb.body.velocity.copy() for rb in world.bodies], axis=0)
    masses = np.array([rb.body.mass for rb in world.bodies], dtype=float).reshape(n)
    static_mask = np.array([rb.body.is_static for rb in world.bodies], dtype=bool).reshape(n)
    inv_m = np.zeros(n, dtype=float)
    inv_m[(~static_mask) & (masses > 0.0)] = 1.0 / masses[(~static_mask) & (masses > 0.0)]

    def set_state(X: np.ndarray, V: np.ndarray) -> None:
        for i, rb in enumerate(world.bodies):
            b = rb.body
            if b.is_static:
                continue
            b.position = X[i].copy()
            b.velocity = V[i].copy()

    def compute_a(X: np.ndarray, V: np.ndarray) -> np.ndarray:
        set_state(X, V)
        F = world.compute_forces()
        a = F * inv_m[:, None]
        return a

    a0 = compute_a(x0, v0)

    alpha_m, alpha_f, gamma, beta = _params(rho_inf)

    a1 = a0.copy()

    for _ in range(max_iter):
        x1 = x0 + dt * v0 + dt * dt * ((0.5 - beta) * a0 + beta * a1)
        v1 = v0 + dt * ((1.0 - gamma) * a0 + gamma * a1)

        Xa = x0 + alpha_f * (x1 - x0)
        Va = v0 + alpha_f * (v1 - v0)
        aa = compute_a(Xa, Va)

        if use_newton:
            denom = alpha_m if alpha_m > 1e-12 else 1e-12
            R_vec = (a1 - a0 - (aa - a0) / denom).reshape(n3)
            if float(np.linalg.norm(R_vec)) < tol:
                break
            dAdX = np.zeros((n3, n3), dtype=float)
            dAdV = np.zeros((n3, n3), dtype=float)
            for i in range(n):
                for k in range(3):
                    idx = i * 3 + k
                    Xa_p = Xa.copy(); Xa_p[i, k] += diff_eps_x
                    aa_p = compute_a(Xa_p, Va).reshape(n3)
                    Xa_m = Xa.copy(); Xa_m[i, k] -= diff_eps_x
                    aa_m = compute_a(Xa_m, Va).reshape(n3)
                    dAdX[:, idx] = (aa_p - aa_m) / (2.0 * diff_eps_x)
                    Va_p = Va.copy(); Va_p[i, k] += diff_eps_v
                    aa_p2 = compute_a(Xa, Va_p).reshape(n3)
                    Va_m = Va.copy(); Va_m[i, k] -= diff_eps_v
                    aa_m2 = compute_a(Xa, Va_m).reshape(n3)
                    dAdV[:, idx] = (aa_p2 - aa_m2) / (2.0 * diff_eps_v)
            scale_x = alpha_f * dt * dt * beta
            scale_v = alpha_f * dt * gamma
            I = np.eye(n3, dtype=float)
            J = I - (1.0 / denom) * (scale_x * dAdX + scale_v * dAdV)
            reg = 1e-10
            J = J + reg * I
            try:
                delta_vec = np.linalg.solve(J, -R_vec)
            except Exception:
                delta_vec = np.linalg.lstsq(J, -R_vec, rcond=None)[0]
            a1 = a1 + delta_vec.reshape(n, 3)
            if float(np.linalg.norm(delta_vec)) < tol:
                break
        else:
            denom = alpha_m if alpha_m > 1e-12 else 1e-12
            a1_new = a0 + (aa - a0) / denom
            diff = a1_new - a1
            a1 = a1 + relax * diff
            if float(np.linalg.norm(diff)) < tol:
                break

    for i, rb in enumerate(world.bodies):
        b = rb.body
        if b.is_static:
            continue
        b.position = x0[i] + dt * v0[i] + dt * dt * ((0.5 - beta) * a0[i] + beta * a1[i])
        b.velocity = v0[i] + dt * ((1.0 - gamma) * a0[i] + gamma * a1[i])


class ContactPerformanceOptimizer:
    """接触计算性能优化器"""
    def __init__(self):
        self.contact_history = []
        self.performance_stats = {
            'contact_detection_time': [],
            'iteration_convergence': [],
            'step_rejection_rate': []
        }
        self.adaptive_params = {
            'min_dt': 1e-6,
            'max_iter': 20,
            'growth_scale': 0.1
        }

    def smart_time_to_collide_prediction(self, world, dt_current):
        """基于历史接触数据的智能碰撞时间预测"""
        if len(self.contact_history) < 3:
            return dt_current * 0.5  # 保守预测
        
        # 分析接触频率和深度变化趋势
        recent_contacts = self.contact_history[-5:]
        if not recent_contacts:
            return dt_current * 0.5
            
        penetration_trend = np.polyfit(range(len(recent_contacts)), 
                                       [c.get('pen', 0) for c in recent_contacts], 1)[0]
        
        if penetration_trend > 0:  # 穿透深度在增加
            return min(dt_current * 0.3, dt_current * 0.9 * recent_contacts[-1].get('ttc', dt_current))
        else:
            return dt_current * 0.8  # 相对宽松的预测
    
    def update_parameters_based_on_performance(self):
        """根据性能统计自适应调整参数"""
        if len(self.performance_stats['step_rejection_rate']) < 5:
            return
            
        recent_rejection_rate = np.mean(self.performance_stats['step_rejection_rate'][-5:])
        
        if recent_rejection_rate > 0.3:  # 步长拒绝率过高
            self.adaptive_params['min_dt'] = min(1e-4, self.adaptive_params['min_dt'] * 1.5)
            self.adaptive_params['growth_scale'] = max(0.01, self.adaptive_params['growth_scale'] * 0.8)
        elif recent_rejection_rate < 0.1:  # 性能良好
            self.adaptive_params['min_dt'] = max(1e-7, self.adaptive_params['min_dt'] * 0.9)


def time_step_backtracking_bisection(world, t_start, dt_initial, max_depth=10, tol=1e-6):
    """
    时间步回退二分法精确计算初始接触时刻
    
    参数:
        world: 物理世界对象
        t_start: 开始时间
        dt_initial: 初始时间步长
        max_depth: 最大二分深度
        tol: 接触时刻精度容差
        
    返回:
        t_contact: 精确的接触时刻
        pen_contact: 接触时刻的穿透深度
        success: 是否成功找到接触时刻
    """
    def capture_state():
        X = np.stack([rb.body.position.copy() for rb in world.bodies], axis=0)
        V = np.stack([rb.body.velocity.copy() for rb in world.bodies], axis=0)
        return X, V

    def restore_state(X, V):
        for i, rb in enumerate(world.bodies):
            b = rb.body
            if b.is_static:
                continue
            b.position = X[i].copy()
            b.velocity = V[i].copy()

    def measure_penetration_safe():
        pen = 0.0
        for pair in world.pairs:
            for cp in pair.manifold.points:
                if cp.phi < 0.0:
                    pen = max(pen, -cp.phi)
        return (pen > 0.0), pen

    def evaluate_contact_at_time(t_eval):
        # 保存当前状态
        X_save, V_save = capture_state()
        
        # 模拟到指定时间
        t_current = t_start
        while t_current < t_eval - 1e-12:
            dt_step = min(dt_initial, t_eval - t_current)
            step_sub(world, dt_step, rho_inf=0.5, max_iter=20, tol=1e-8)
            t_current += dt_step
        
        # 测量接触状态
        in_contact, penetration = measure_penetration_safe()
        
        # 恢复状态
        restore_state(X_save, V_save)
        
        return in_contact, penetration
    
    # 检查初始时间点是否已经接触
    in_contact_start, pen_start = evaluate_contact_at_time(t_start)
    if in_contact_start:
        return t_start, pen_start, True
    
    # 检查结束时间点是否接触
    t_end = t_start + dt_initial
    in_contact_end, pen_end = evaluate_contact_at_time(t_end)
    if not in_contact_end:
        # 在给定时间范围内没有接触发生
        return t_end, 0.0, False
    
    # 二分法搜索接触时刻
    t_low = t_start
    t_high = t_end
    
    for depth in range(max_depth):
        t_mid = (t_low + t_high) / 2.0
        in_contact_mid, pen_mid = evaluate_contact_at_time(t_mid)
        
        if in_contact_mid:
            t_high = t_mid
        else:
            t_low = t_mid
        
        # 检查收敛条件
        if (t_high - t_low) < tol:
            # 返回接触时刻和穿透深度
            return t_high, pen_mid, True
    
    # 达到最大深度，返回最佳估计
    return t_high, pen_end, True


def simulate(
    world: ContactWorld,
    t_end: float,
    dt_frame: float,
    dt_sub: float,
    on_frame: Optional[Callable[[int, float, ContactWorld], None]] = None,
    on_substep: Optional[Callable[[float, ContactWorld], None]] = None,
    *,
    rho_inf: float = 0.5,
    max_iter: int = 20,
    tol: float = 1e-8,
    relax: float = 1.0,
    use_newton: bool = False,
    diff_eps_x: float = 1e-6,
    diff_eps_v: float = 1e-6,
    contact_dt_cap: float = 1.0e-3,
    min_dt: float = 1.0e-6,
    contact_growth_thr: float = 1.0e-3,
    contact_steady_tol: float = 1.0e-4,
    init_thr_scale: float = 1.0,
    init_thr_max: float = 5.0e-3,
    growth_scale: float = 0.1,
    contact_grace_steps: int = 1,
    use_performance_optimization: bool = True,  # 新增：启用性能优化
) -> None:

    def predict_time_to_collide() -> tuple[bool, float]:
        def swept_aabb_ttc(boxA: Aabb, vA: np.ndarray, boxB: Aabb, vB: np.ndarray) -> float | None:
            vrel = np.asarray(vA - vB, float).reshape(3)
            t_enter = -float('inf')
            t_exit = float('inf')
            for k in range(3):
                Amin = float(boxA.m_min[k])
                Amax = float(boxA.m_max[k])
                Bmin = float(boxB.m_min[k])
                Bmax = float(boxB.m_max[k])
                vd = float(vrel[k])
                if abs(vd) < 1e-12:
                    if (Amax < Bmin) or (Bmax < Amin):
                        return None
                    te = -float('inf')
                    tx = float('inf')
                else:
                    inv = 1.0 / vd
                    if vd > 0.0:
                        te = (Bmin - Amax) * inv
                        tx = (Bmax - Amin) * inv
                    else:
                        te = (Bmax - Amin) * inv
                        tx = (Bmin - Amax) * inv
                if te > t_enter:
                    t_enter = te
                if tx < t_exit:
                    t_exit = tx
            if (t_enter <= t_exit) and (t_exit >= 0.0):
                return max(0.0, t_enter)
            return None

        imminent = False
        ttc_min = float('inf')
        for pair in world.pairs:
            i, j = pair.i, pair.j
            A = world.bodies[i]
            B = world.bodies[j]
            boxA = Aabb()
            boxA.m_min = A.mesh.aabb.m_min.copy()
            boxA.m_max = A.mesh.aabb.m_max.copy()
            boxA.update((A.body.position, A.R))
            boxB = Aabb()
            boxB.m_min = B.mesh.aabb.m_min.copy()
            boxB.m_max = B.mesh.aabb.m_max.copy()
            boxB.update((B.body.position, B.R))
            ttc = swept_aabb_ttc(boxA, A.body.velocity, boxB, B.body.velocity)
            if ttc is not None:
                if ttc < ttc_min:
                    ttc_min = ttc
                    imminent = True
        return imminent, (ttc_min if imminent else float('inf'))

    def capture_state():
        X = np.stack([rb.body.position.copy() for rb in world.bodies], axis=0)
        V = np.stack([rb.body.velocity.copy() for rb in world.bodies], axis=0)
        return X, V

    def restore_state(X, V):
        for i, rb in enumerate(world.bodies):
            b = rb.body
            if b.is_static:
                continue
            b.position = X[i].copy()
            b.velocity = V[i].copy()

    def measure_penetration() -> tuple[bool, float]:
        pen = 0.0
        for pair in world.pairs:
            for cp in pair.manifold.points:
                if cp.phi < 0.0:
                    pen = max(pen, -cp.phi)
        return (pen > 0.0), pen

    def measure_penetration_intersect_only() -> tuple[bool, float]:
        pen = 0.0
        for pair in world.pairs:
            i, j = pair.i, pair.j
            A = world.bodies[i]
            B = world.bodies[j]
            poseA = (A.body.position, A.R)
            poseB = (B.body.position, B.R)
            pairs_idx: list[tuple[int, int]] = []
            pair.detector.meshA.bvh.find_collision(pair.detector.meshB.bvh, poseA, poseB, pairs_idx)
            if not pairs_idx:
                continue
            for ia, ib in pairs_idx[:1000]:
                nodesA = A.mesh.get_triangle_nodes(ia, poseA)
                nodesB = B.mesh.get_triangle_nodes(ib, poseB)
                res = CollisionResult()
                triTriIntersect(nodesA, nodesB, res)
                if res.contPtsPairs:
                    for pA_world, pB_world in res.contPtsPairs:
                        v = pB_world - pA_world
                        dist = float(np.linalg.norm(v))
                        if dist > pen:
                            pen = dist
        return (pen > 0.0), pen

    def measure_penetration_safe() -> tuple[bool, float]:
        ic1, p1 = measure_penetration()
        if ic1:
            return ic1, p1
        ic2, p2 = measure_penetration_intersect_only()
        if ic2:
            return True, max(p1, p2)
        return False, 0.0
    
    t = 0.0
    frame = 0
    
    contact_occurrence_flag = False
    continuous_contact_flag = False
    prev_pen = 0.0
    
    contact_state_history = []
    max_history_length = 20
    
    continuous_contact_threshold = float(contact_growth_thr)
    pre_contact_eps = 1e-6
    steady_tol = float(contact_steady_tol)
    min_dt_val = float(min_dt)
    dt_current = dt_sub
    grace_steps_remaining = 0
    min_contact_dt = max(min_dt_val, float(contact_dt_cap) / 16.0)
    
    optimizer = ContactPerformanceOptimizer() if use_performance_optimization else None
    
    if use_performance_optimization and optimizer is not None:
        min_dt_val = optimizer.adaptive_params['min_dt']
        max_iter = optimizer.adaptive_params['max_iter']
        growth_scale = optimizer.adaptive_params['growth_scale']
    
    dt_relaxation_factor = 1.1
    dt_relaxation_steps = 5
    dt_relaxation_counter = 0
    dt_target = dt_sub
    
    while t < t_end - 1e-12:
        try:
            perf.set_meta(algorithm="genalpha", t_end=t_end, dt_frame=dt_frame, dt_sub=dt_sub, bodies=len(world.bodies))
        except Exception:
            pass
        remaining = dt_frame
        while remaining > 1e-12:
            dt_try = min(dt_current, remaining)
            contact_prev = (contact_occurrence_flag or continuous_contact_flag)
            vmax = float(np.max([float(np.linalg.norm(rb.body.velocity)) for rb in world.bodies if not rb.body.is_static] + [0.0]))
            move_est = vmax * dt_try
            pre_eps = max(1e-6, 0.1 * move_est)
            init_thr = min(1.0e-3, max(1e-6, 0.1 * move_est))
            min_phi_preview = float("inf")
            vn_min_preview = 0.0
            for pair in world.pairs:
                ia, ja = pair.i, pair.j
                A = world.bodies[ia]
                B = world.bodies[ja]
                poseA = (A.body.position, A.R)
                poseB = (B.body.position, B.R)
                try:
                    near_preview = move_est + 0.5 * (dt_try ** 2) * abs(world.g)
                    pair.detector._near_band = 0.5 * float(near_preview)
                except Exception:
                    pair.detector._near_band = float(move_est)
                manifold = pair.detector.query_manifold(poseA, poseB, prev_manifold=pair.manifold, max_points=4)
                pair.manifold = manifold
                rel_vp = A.body.velocity - B.body.velocity
                for cp in manifold.points:
                    Lp = float(np.linalg.norm(cp.n))
                    nvecp = np.array([0.0, 0.0, 1.0], dtype=float) if Lp < 1.0e-12 else (cp.n / Lp)
                    if cp.phi < min_phi_preview:
                        min_phi_preview = cp.phi
                    vnp = float(np.dot(rel_vp, nvecp))
                    if vnp < vn_min_preview:
                        vn_min_preview = vnp
                if manifold.is_empty():
                    cA = A.R @ A.mesh.center_local + A.body.position
                    cB = B.R @ B.mesh.center_local + B.body.position
                    dvec = cB - cA
                    dlen = float(np.linalg.norm(dvec))
                    ddir = np.array([0.0, 0.0, 1.0], float) if dlen < 1e-12 else (dvec / dlen)
                    trisA = A.mesh.triangles
                    trisB = B.mesh.triangles
                    nA = len(trisA)
                    nB = len(trisB)
                    strideA = max(1, nA // 64)
                    strideB = max(1, nB // 64)
                    candA = []
                    candB = []
                    for idx in range(0, nA, strideA):
                        tri = trisA[idx]
                        nodes = [A.R @ A.mesh.vertices[tri[k]] + A.body.position for k in range(3)]
                        cen = (nodes[0] + nodes[1] + nodes[2]) / 3.0
                        score = float(np.dot(cen, ddir))
                        candA.append((idx, score))
                    for idx in range(0, nB, strideB):
                        tri = trisB[idx]
                        nodes = [B.R @ B.mesh.vertices[tri[k]] + B.body.position for k in range(3)]
                        cen = (nodes[0] + nodes[1] + nodes[2]) / 3.0
                        score = float(np.dot(cen, ddir))
                        candB.append((idx, score))
                    candA.sort(key=lambda x: x[1], reverse=True)
                    candB.sort(key=lambda x: x[1])
                    selA = [i for i, _ in candA[:8]]
                    selB = [i for i, _ in candB[:8]]
                    for ia2 in selA:
                        nodesA = A.mesh.get_triangle_nodes(ia2, poseA)
                        for ib2 in selB:
                            nodesB = B.mesh.get_triangle_nodes(ib2, poseB)
                            dist, _, _, nBv = triangle_distance(nodesA, nodesB)
                            if dist < min_phi_preview:
                                min_phi_preview = dist
                            vn_preview = float(np.dot(rel_vp, nBv))
                            if vn_preview < vn_min_preview:
                                vn_min_preview = vn_preview
            pre_capture_hint = (min_phi_preview <= pre_eps) and (vn_min_preview < 0.0)
            if (not (contact_occurrence_flag or continuous_contact_flag)) and pre_capture_hint:
                if min_phi_preview > 0.0 and vn_min_preview < 0.0:
                    ttc_est = min_phi_preview / (-vn_min_preview)
                    if use_performance_optimization and optimizer is not None:
                        dt_try = optimizer.smart_time_to_collide_prediction(world, dt_try)
                    else:
                        dt_try = min(dt_try, max(min_dt_val, 0.9 * ttc_est))
                else:
                    dt_try = max(min_dt_val, 0.5 * dt_try)
            if contact_occurrence_flag or continuous_contact_flag:
                dt_try = min(dt_try, float(contact_dt_cap))
            X0, V0 = capture_state()
            world._step_max_move = dt_try * float(np.max([float(np.linalg.norm(rb.body.velocity)) for rb in world.bodies if not rb.body.is_static] + [0.0])) + 0.5 * (dt_try ** 2) * abs(world.g)
            world._force_fresh_manifold = (not (contact_occurrence_flag or continuous_contact_flag))
            with perf.section("integrator.step_sub"):
                step_sub(world, dt_try, rho_inf=rho_inf, max_iter=max_iter, tol=tol, relax=relax, use_newton=use_newton, diff_eps_x=diff_eps_x, diff_eps_v=diff_eps_v)
            in_contact, pen = measure_penetration_safe()
            accepted_in_contact = in_contact
            min_phi = float("inf")
            approaching = False
            for pair in world.pairs:
                ia, ja = pair.i, pair.j
                rel_v2 = world.bodies[ia].body.velocity - world.bodies[ja].body.velocity
                for cp in pair.manifold.points:
                    L = float(np.linalg.norm(cp.n))
                    nvec = np.array([0.0, 0.0, 1.0], dtype=float) if L < 1e-12 else (cp.n / L)
                    if cp.phi < min_phi:
                        min_phi = cp.phi
                    vn2 = float(np.dot(rel_v2, nvec))
                    if vn2 < 0.0:
                        approaching = True
                        if min_phi <= pre_contact_eps:
                            break
            pre_capture = (min_phi <= pre_eps) and approaching
            accept = True
            target_dt = dt_try
            initial_phase = False
            if continuous_contact_flag:
                if pen > prev_pen:
                    inc = pen - prev_pen
                    if inc > continuous_contact_threshold:
                        accept = False
                else:
                    if abs(pen - prev_pen) < steady_tol:
                        dt_current = min(dt_sub, dt_current * 2.0)
            elif contact_occurrence_flag:
                if in_contact:
                    if pen > init_thr:
                        initial_phase = True
                        accept = False
                        prev_pen = 0.0
                    else:
                        continuous_contact_flag = True
                        contact_occurrence_flag = False
                else:
                    contact_occurrence_flag = False
            else:
                if in_contact:
                    contact_occurrence_flag = True
                    if pen > init_thr:
                        initial_phase = True
                        accept = False
                        prev_pen = 0.0
                        t_contact, pen_contact, success = time_step_backtracking_bisection(
                            world, t, dt_try, max_depth=8, tol=1e-6
                        )
                        if success:
                            restore_state(X0, V0)
                            dt_exact = t_contact - t
                            if dt_exact > min_dt_val:
                                world._step_max_move = dt_exact * float(np.max([float(np.linalg.norm(rb.body.velocity)) for rb in world.bodies if not rb.body.is_static] + [0.0])) + 0.5 * (dt_exact ** 2) * abs(world.g)
                                world._force_fresh_manifold = True
                                with perf.section("integrator.step_sub"):
                                    step_sub(world, dt_exact, rho_inf=rho_inf, max_iter=max_iter, tol=tol, relax=relax, use_newton=use_newton, diff_eps_x=diff_eps_x, diff_eps_v=diff_eps_v)
                                t += dt_exact
                                remaining -= dt_exact
                                in_contact, pen = measure_penetration_safe()
                                contact_occurrence_flag = True
                                continuous_contact_flag = False
                                dt_relaxation_counter = 0
                                dt_current = min(dt_exact, dt_target)
                                if use_performance_optimization and optimizer is not None:
                                    contact_info = {
                                        't': t,
                                        'contact_occurrence_flag': contact_occurrence_flag,
                                        'continuous_contact_flag': continuous_contact_flag,
                                        'pen': pen,
                                        'in_contact': in_contact,
                                        'exact_contact_time': True,
                                        'dt_exact': dt_exact
                                    }
                                    optimizer.contact_history.append(contact_info)
                elif pre_capture:
                    initial_phase = True
                    accept = False
                    prev_pen = 0.0
            if (contact_occurrence_flag or continuous_contact_flag) and (grace_steps_remaining > 0):
                accept = True
                target_dt = dt_try
                accepted_in_contact = in_contact
                grace_steps_remaining -= 1
            if not accept:
                restore_state(X0, V0)
                low = min_contact_dt if (initial_phase or (contact_occurrence_flag or continuous_contact_flag)) else float(min_dt_val)
                dt_try2 = max(low, 0.5 * dt_try)
                if use_performance_optimization and optimizer is not None:
                    optimizer.performance_stats['step_rejection_rate'].append(1.0)
                    if len(optimizer.performance_stats['step_rejection_rate']) % 10 == 0:
                        optimizer.update_parameters_based_on_performance()
                        min_dt_val = optimizer.adaptive_params['min_dt']
                        growth_scale = optimizer.adaptive_params['growth_scale']
                world._step_max_move = dt_try2 * float(np.max([float(np.linalg.norm(rb.body.velocity)) for rb in world.bodies if not rb.body.is_static] + [0.0])) + 0.5 * (dt_try2 ** 2) * abs(world.g)
                world._force_fresh_manifold = True
                with perf.section("integrator.step_sub"):
                    step_sub(world, dt_try2, rho_inf=rho_inf, max_iter=max_iter, tol=tol, relax=relax, use_newton=use_newton, diff_eps_x=diff_eps_x, diff_eps_v=diff_eps_v)
                in_contact2, pen2 = measure_penetration_safe()
                if initial_phase:
                    vmax2 = float(np.max([float(np.linalg.norm(rb.body.velocity)) for rb in world.bodies if not rb.body.is_static] + [0.0]))
                    thr2 = min(float(init_thr_max), max(low, float(init_thr_scale) * vmax2 * dt_try2))
                    if pen2 <= thr2:
                        target_dt = dt_try2
                        pen = pen2
                        contact_occurrence_flag = True
                        accepted_in_contact = in_contact2
                    else:
                        dt_iter = dt_try2
                        Xb, Vb = X0, V0
                        pen_iter = pen2
                        in_contact_iter = in_contact2
                        while True:
                            if dt_iter <= low + 1e-12:
                                break
                            restore_state(Xb, Vb)
                            dt_iter = max(low, 0.5 * dt_iter)
                            world._step_max_move = dt_iter * float(np.max([float(np.linalg.norm(rb.body.velocity)) for rb in world.bodies if not rb.body.is_static] + [0.0])) + 0.5 * (dt_iter ** 2) * abs(world.g)
                            world._force_fresh_manifold = True
                            with perf.section("integrator.step_sub"):
                                step_sub(world, dt_iter, rho_inf=rho_inf, max_iter=max_iter, tol=tol, relax=relax, use_newton=use_newton, diff_eps_x=diff_eps_x, diff_eps_v=diff_eps_v)
                            vmax3 = float(np.max([float(np.linalg.norm(rb.body.velocity)) for rb in world.bodies if not rb.body.is_static] + [0.0]))
                            thr3 = min(float(init_thr_max), max(low, float(init_thr_scale) * vmax3 * dt_iter))
                            in_contact_iter, pen_iter = measure_penetration_safe()
                            if pen_iter <= thr3:
                                break
                        target_dt = dt_iter
                        pen = pen_iter if pen_iter <= thr2 else 0.0
                        if pen_iter <= thr2:
                            contact_occurrence_flag = True
                            accepted_in_contact = in_contact_iter
                        else:
                            contact_occurrence_flag = False
                            accepted_in_contact = False
                else:
                    if continuous_contact_flag:
                        inc2 = pen2 - prev_pen
                        vmaxc = float(np.max([float(np.linalg.norm(rb.body.velocity)) for rb in world.bodies if not rb.body.is_static] + [0.0]))
                        thr_growth = max(continuous_contact_threshold, float(growth_scale) * vmaxc * dt_try2)
                        if inc2 <= thr_growth:
                            target_dt = dt_try2
                            pen = pen2
                            accepted_in_contact = in_contact2
                        else:
                            dt_iter = dt_try2
                            Xb, Vb = X0, V0
                            while True:
                                if dt_iter <= low + 1e-12:
                                    break
                                restore_state(Xb, Vb)
                                dt_iter = max(low, 0.5 * dt_iter)
                                world._step_max_move = dt_iter * float(np.max([float(np.linalg.norm(rb.body.velocity)) for rb in world.bodies if not rb.body.is_static] + [0.0])) + 0.5 * (dt_iter ** 2) * abs(world.g)
                                world._force_fresh_manifold = True
                                with perf.section("integrator.step_sub"):
                                    step_sub(world, dt_iter, rho_inf=rho_inf, max_iter=max_iter, tol=tol, relax=relax, use_newton=use_newton, diff_eps_x=diff_eps_x, diff_eps_v=diff_eps_v)
                                in_contact_iter, pen_iter = measure_penetration()
                                vmaxi = float(np.max([float(np.linalg.norm(rb.body.velocity)) for rb in world.bodies if not rb.body.is_static] + [0.0]))
                                thr_growth_i = max(continuous_contact_threshold, float(growth_scale) * vmaxi * dt_iter)
                                inc_iter = pen_iter - prev_pen
                                if inc_iter <= thr_growth_i:
                                    break
                            target_dt = dt_iter
                            if target_dt > low + 1e-12:
                                pen = pen_iter
                                accepted_in_contact = in_contact_iter
                            else:
                                pen = 0.0
                                contact_occurrence_flag = False
                                continuous_contact_flag = False
                                accepted_in_contact = False
                world._force_fresh_manifold = False
                in_contact = accepted_in_contact
                if not in_contact:
                    contact_occurrence_flag = False
                    continuous_contact_flag = False
                elif contact_occurrence_flag and pen <= init_thr:
                    continuous_contact_flag = True
                    contact_occurrence_flag = False
            if (contact_occurrence_flag or continuous_contact_flag) and (not in_contact):
                contact_occurrence_flag = False
                continuous_contact_flag = False
                if dt_current < dt_target:
                    if prev_pen > 0.01:
                        dt_current = min(dt_target, dt_current * 2.0)
                    elif prev_pen > 0.001:
                        dt_current = min(dt_target, dt_current * 1.5)
                    else:
                        dt_current = min(dt_target, dt_current * 1.2)
                    dt_relaxation_counter = 0
                if use_performance_optimization and optimizer is not None:
                    separation_info = {
                        't': t,
                        'type': 'separation',
                        'prev_pen': prev_pen,
                        'dt_current': dt_current,
                        'recovery_factor': 2.0 if prev_pen > 0.01 else (1.5 if prev_pen > 0.001 else 1.2)
                    }
                    optimizer.contact_history.append(separation_info)
            prev_pen = pen if in_contact or contact_occurrence_flag or continuous_contact_flag else 0.0
            remaining -= target_dt
            t += target_dt
            if continuous_contact_flag and dt_current < dt_target:
                dt_relaxation_counter += 1
                if dt_relaxation_counter >= dt_relaxation_steps:
                    dt_current = min(dt_target, dt_current * dt_relaxation_factor)
                    dt_relaxation_counter = 0
            elif not (contact_occurrence_flag or continuous_contact_flag):
                dt_relaxation_counter = 0
                dt_current = dt_target
            if use_performance_optimization and optimizer is not None:
                optimizer.performance_stats['step_rejection_rate'].append(0.0)
                contact_info = {
                    't': t,
                    'pen': pen,
                    'dt': target_dt,
                    'in_contact': in_contact
                }
                optimizer.contact_history.append(contact_info)
                if len(optimizer.contact_history) > 50:
                    optimizer.contact_history.pop(0)
            if contact_occurrence_flag or continuous_contact_flag:
                dt_current = min(dt_sub, float(contact_dt_cap))
            else:
                dt_current = min(dt_sub, dt_current * 2.0)
            contact_state_history.append({
                't': t,
                'contact_occurrence_flag': contact_occurrence_flag,
                'continuous_contact_flag': continuous_contact_flag,
                'pen': pen,
                'in_contact': in_contact
            })
            if len(contact_state_history) > max_history_length:
                contact_state_history.pop(0)
            contact_prev = (contact_occurrence_flag or continuous_contact_flag)
            if (not contact_prev) and in_contact:
                try:
                    grace_steps_remaining = int(contact_grace_steps)
                except Exception:
                    grace_steps_remaining = 1
            if (prev_pen <= steady_tol) and in_contact and (grace_steps_remaining == 0):
                try:
                    grace_steps_remaining = int(contact_grace_steps)
                except Exception:
                    grace_steps_remaining = 1
            if on_substep is not None:
                with perf.section("callback.on_substep"):
                    on_substep(t, world)
        if on_frame is not None:
            with perf.section("callback.on_frame"):
                on_frame(frame, t, world)
        frame += 1
