import numpy as np
from typing import List, Tuple
import math

from .contact_types import CollisionResult, ContactType, FLT_EPSILON, EILON_2


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v.copy()


def _point_triangle_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> Tuple[float, np.ndarray]:
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = float(np.dot(ab, ap))
    d2 = float(np.dot(ac, ap))
    if d1 <= 0 and d2 <= 0:
        return np.linalg.norm(p - a), a
    bp = p - b
    d3 = float(np.dot(ab, bp))
    d4 = float(np.dot(ac, bp))
    if d3 >= 0 and d4 <= d3:
        return np.linalg.norm(p - b), b
    vc = d1 * d4 - d3 * d2
    if vc <= 0 and d1 >= 0 and d3 <= 0:
        v = d1 / (d1 - d3)
        proj = a + v * ab
        return np.linalg.norm(p - proj), proj
    cp = p - c
    d5 = float(np.dot(ab, cp))
    d6 = float(np.dot(ac, cp))
    if d6 >= 0 and d5 <= d6:
        return np.linalg.norm(p - c), c
    vb = d5 * d2 - d1 * d6
    if vb <= 0 and d2 >= 0 and d6 <= 0:
        w = d2 / (d2 - d6)
        proj = a + w * ac
        return np.linalg.norm(p - proj), proj
    va = d3 * d6 - d5 * d4
    if va <= 0 and (d4 - d3) >= 0 and (d5 - d6) >= 0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        proj = b + w * (c - b)
        return np.linalg.norm(p - proj), proj
    n = np.cross(ab, ac)
    n = _normalize(n)
    dist = float(np.dot(p - a, n))
    proj = p - dist * n
    return abs(dist), proj


def triangle_distance(a_tri: List[np.ndarray], b_tri: List[np.ndarray]) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    min_d = np.inf
    best_a = np.zeros(3)
    best_b = np.zeros(3)
    best_nB = np.array([0.0, 0.0, 1.0])
    for i in range(3):
        a0 = a_tri[i]
        a1 = a_tri[(i + 1) % 3]
        for j in range(3):
            b0 = b_tri[j]
            b1 = b_tri[(j + 1) % 3]
            edgeA = a1 - a0
            edgeB = b1 - b0
            w0 = a0 - b0
            a = np.dot(edgeA, edgeA)
            b_val = np.dot(edgeA, edgeB)
            c = np.dot(edgeB, edgeB)
            d = np.dot(edgeA, w0)
            e = np.dot(edgeB, w0)
            denom = a * c - b_val * b_val
            s, t = 0.0, 0.0
            if abs(denom) > 1e-12:
                s = (b_val * e - c * d) / denom
                t = (a * e - b_val * d) / denom
                s = max(0, min(1, s))
                t = max(0, min(1, t))
            else:
                s = 0
                t = d / b_val if abs(b_val) > 1e-12 else e / c
                t = max(0, min(1, t))
            closest_A = a0 + s * edgeA
            closest_B = b0 + t * edgeB
            dist = np.linalg.norm(closest_B - closest_A)
            if dist < min_d:
                min_d = dist
                best_a = closest_A
                best_b = closest_B
                if dist > 1e-12:
                    best_nB = (closest_B - closest_A) / dist
    for i in range(3):
        point = a_tri[i]
        dist, proj = _point_triangle_distance(point, b_tri[0], b_tri[1], b_tri[2])
        if dist < min_d:
            min_d = dist
            best_a = point
            best_b = proj
            if dist > 1e-12:
                best_nB = (proj - point) / dist
    for i in range(3):
        point = b_tri[i]
        dist, proj = _point_triangle_distance(point, a_tri[0], a_tri[1], a_tri[2])
        if dist < min_d:
            min_d = dist
            best_a = proj
            best_b = point
            if dist > 1e-12:
                best_nB = (point - proj) / dist
    return float(min_d), best_a, best_b, best_nB


def SegmentDistance2(nodesA: List[np.ndarray], nodesB: List[np.ndarray], clstPntA: np.ndarray, clstPntB: np.ndarray, vec: np.ndarray) -> None:
    edgeA = nodesA[1] - nodesA[0]
    edgeB = nodesB[1] - nodesB[0]
    edgeC = nodesB[0] - nodesA[0]
    AdotA = float(np.dot(edgeA, edgeA))
    BdotB = float(np.dot(edgeB, edgeB))
    AdotB = float(np.dot(edgeA, edgeB))
    AdotC = float(np.dot(edgeA, edgeC))
    BdotC = float(np.dot(edgeB, edgeC))
    denom = AdotA * BdotB - AdotB * AdotB
    t = (AdotC * BdotB - BdotC * AdotB) / denom if denom != 0 else float('nan')
    if t < 0 or np.isnan(t):
        t = 0.0
    elif t > 1:
        t = 1.0
    u = (t * AdotB - BdotC) / BdotB if BdotB != 0 else float('nan')
    if u <= 0 or np.isnan(u):
        clstPntB[:] = nodesB[0]
        t2 = AdotC / AdotA if AdotA != 0 else float('nan')
        if t2 <= 0 or np.isnan(t2):
            clstPntA[:] = nodesA[0]
            vec[:] = clstPntB - clstPntA
        elif t2 >= 1:
            clstPntA[:] = nodesA[1]
            vec[:] = clstPntB - clstPntA
        else:
            clstPntA[:] = nodesA[0] + t2 * edgeA
            tmp = np.cross(edgeC, edgeA)
            vec[:] = np.cross(edgeA, tmp)
    elif u >= 1:
        clstPntB[:] = nodesB[1]
        t2 = (AdotB + AdotC) / AdotA if AdotA != 0 else float('nan')
        if t2 <= 0 or np.isnan(t2):
            clstPntA[:] = nodesA[0]
            vec[:] = clstPntB - clstPntA
        elif t2 >= 1:
            clstPntA[:] = nodesA[1]
            vec[:] = clstPntB - clstPntA
        else:
            clstPntA[:] = nodesA[0] + t2 * edgeA
            tmp = np.cross(clstPntB - nodesA[0], edgeA)
            vec[:] = np.cross(edgeA, tmp)
    else:
        clstPntB[:] = nodesB[0] + u * edgeB
        if t <= 0 or np.isnan(t):
            clstPntA[:] = nodesA[0]
            tmp = np.cross(edgeC, edgeB)
            vec[:] = np.cross(edgeB, tmp)
        elif t >= 1:
            clstPntA[:] = nodesA[1]
            tmp = np.cross(nodesB[0] - clstPntA, edgeB)
            vec[:] = np.cross(edgeB, tmp)
        else:
            clstPntA[:] = nodesA[0] + t * edgeA
            vec[:] = np.cross(edgeA, edgeB)
            if float(np.dot(vec, edgeC)) < 0.0:
                vec[:] = -vec


def isect2(VTX0: np.ndarray, VTX1: np.ndarray, VTX2: np.ndarray, VV0: float, VV1: float, VV2: float, D0: float, D1: float, D2: float):
    tmp = D0 / (D0 - D1) if D0 - D1 != 0 else 0.0
    isect0 = VV0 + (VV1 - VV0) * tmp
    diff = (VTX1 - VTX0) * tmp
    isectPoint0 = VTX0 + diff
    tmp = D0 / (D0 - D2) if D0 - D2 != 0 else 0.0
    isect1 = VV0 + (VV2 - VV0) * tmp
    diff = (VTX2 - VTX0) * tmp
    isectPoint1 = VTX0 + diff
    return (isect0, isect1, isectPoint0, isectPoint1)


def computeIntervalsIsectline(VERT0: np.ndarray, VERT1: np.ndarray, VERT2: np.ndarray, VV0: float, VV1: float, VV2: float, D0: float, D1: float, D2: float, D0D1: float, D0D2: float):
    if D0D1 > FLT_EPSILON:
        return (0, *isect2(VERT2, VERT0, VERT1, VV2, VV0, VV1, D2, D0, D1))
    elif D0D2 > FLT_EPSILON:
        return (0, *isect2(VERT1, VERT0, VERT2, VV1, VV0, VV2, D1, D0, D2))
    elif D1 * D2 > FLT_EPSILON or D0 != 0.0:
        return (0, *isect2(VERT0, VERT1, VERT2, VV0, VV1, VV2, D0, D1, D2))
    elif D1 != 0.0:
        return (0, *isect2(VERT1, VERT0, VERT2, VV1, VV0, VV2, D1, D0, D2))
    elif D2 != 0.0:
        return (0, *isect2(VERT2, VERT0, VERT1, VV2, VV0, VV1, D2, D0, D1))
    else:
        return (1, 0, 0, np.zeros(3), np.zeros(3))


def computeIntersectionPoints(isect1: List[float], isect2_: List[float], isectpointA1: np.ndarray, isectpointA2: np.ndarray, isectpointB1: np.ndarray, isectpointB2: np.ndarray, result: CollisionResult) -> None:
    if isect2_[0] < isect1[0]:
        result.isectpt1 = isectpointA1
        if isect2_[1] < isect1[1]:
            result.isectpt2 = isectpointB2
        else:
            result.isectpt2 = isectpointA2
    else:
        result.isectpt1 = isectpointB1
        if isect2_[1] > isect1[1]:
            result.isectpt2 = isectpointA2
        else:
            result.isectpt2 = isectpointB2


def isInsideTriangle(nodesP: np.ndarray, nodesB: List[np.ndarray]) -> bool:
    edgeAB = nodesB[1] - nodesB[0]
    edgeAC = nodesB[2] - nodesB[0]
    area = 0.5 * np.linalg.norm(np.cross(edgeAB, edgeAC))
    Sabp = 0.5 * np.linalg.norm(np.cross(nodesB[1] - nodesB[0], nodesP - nodesB[0]))
    Sacp = 0.5 * np.linalg.norm(np.cross(nodesB[2] - nodesB[0], nodesP - nodesB[0]))
    Sbcp = 0.5 * np.linalg.norm(np.cross(nodesB[2] - nodesB[1], nodesP - nodesB[1]))
    error = abs(area - (Sabp + Sacp + Sbcp))
    return error <= FLT_EPSILON


def coplanarTriTriIntersect(nodesA: List[np.ndarray], nodesB: List[np.ndarray], result: CollisionResult) -> None:
    for i in range(3):
        edgeA = [nodesA[i], nodesA[(i + 1) % 3]]
        for j in range(3):
            clstPntA = np.zeros(3)
            clstPntB = np.zeros(3)
            vec = np.zeros(3)
            edgeB = [nodesB[j], nodesB[(j + 1) % 3]]
            SegmentDistance2(edgeA, edgeB, clstPntA, clstPntB, vec)
            if np.linalg.norm(clstPntA - clstPntB) < FLT_EPSILON:
                result.contPtsPairs.append((clstPntA, clstPntB))
    if len(result.contPtsPairs) < 1:
        result.contactResult = ContactType.separate
    else:
        result.contactResult = ContactType.surfaceToSurface


def TriangleDistance2(nodesA: List[np.ndarray], nodesB: List[np.ndarray], result: CollisionResult) -> None:
    A0, A1, A2 = nodesA
    B0, B1, B2 = nodesB
    clstPntA = np.zeros(3)
    clstPntB = np.zeros(3)
    disjoint = False
    minDist2 = np.linalg.norm(A0) + 1.0
    vec = np.zeros(3)
    for ii in range(3):
        edgeA = [nodesA[ii], nodesA[(ii + 1) % 3]]
        for jj in range(3):
            edgeB = [nodesB[jj], nodesB[(jj + 1) % 3]]
            tmpA = np.zeros(3)
            tmpB = np.zeros(3)
            SegmentDistance2(edgeA, edgeB, tmpA, tmpB, vec)
            v = tmpB - tmpA
            dist2 = float(np.dot(v, v))
            if dist2 < minDist2:
                clstPntA[:] = tmpA
                clstPntB[:] = tmpB
                minDist2 = dist2
                slab = tmpB - tmpA
                a = float(np.dot(nodesA[(ii + 2) % 3] - tmpA, vec))
                b = float(np.dot(nodesB[(jj + 2) % 3] - tmpB, vec))
                c = float(np.dot(slab, vec))
                if a <= 0 and b >= 0:
                    result.contactResult = ContactType.separate
                    result.closetPtsPairs.append((clstPntA.copy(), clstPntB.copy()))
                    return
                if a < 0:
                    a = 0.0
                if b > 0:
                    b = 0.0
                tmpc = c - a + b
                if tmpc > 0:
                    disjoint = True
    edgesA = [nodesA[1] - nodesA[0], nodesA[2] - nodesA[1], nodesA[0] - nodesA[2]]
    normalA = np.cross(edgesA[0], edgesA[1])
    edgesB = [nodesB[1] - nodesB[0], nodesB[2] - nodesB[1], nodesB[0] - nodesB[2]]
    normalB = np.cross(edgesB[0], edgesB[1])
    lenNormalB2 = float(np.dot(normalB, normalB))
    if lenNormalB2 > EILON_2:
        prjA = [float(np.dot(nodesA[k] - B0, normalB)) for k in range(3)]
        clstId = -1
        if prjA[0] > 0 and prjA[1] > 0 and prjA[2] > 0:
            clstId = 0 if prjA[0] < prjA[1] else 1
            clstId = 2 if prjA[2] < prjA[clstId] else clstId
        elif prjA[0] < 0 and prjA[1] < 0 and prjA[2] < 0:
            clstId = 0 if prjA[0] > prjA[1] else 1
            clstId = 2 if prjA[2] > prjA[clstId] else clstId
        if clstId >= 0:
            ii = 0
            for ii in range(3):
                outter = np.cross(edgesB[ii], normalB)
                if float(np.dot(outter, nodesA[clstId] - nodesB[ii])) > 0:
                    break
            if ii == 3 or ii == 2 and False:
                clstPntA = nodesA[clstId]
                dist = float(prjA[clstId]) / (math.sqrt(lenNormalB2) if lenNormalB2 > 0 else 1.0)
                clstPntB = clstPntA - (normalB / (math.sqrt(lenNormalB2) if lenNormalB2 > 0 else 1.0)) * dist
                result.contactResult = ContactType.separate
                result.closetPtsPairs.append((clstPntA.copy(), clstPntB.copy()))
                return
    lenNormalA2 = float(np.dot(normalA, normalA))
    if lenNormalA2 > EILON_2:
        prjB = [float(np.dot(nodesB[k] - A0, normalA)) for k in range(3)]
        clstId = -1
        if prjB[0] > 0 and prjB[1] > 0 and prjB[2] > 0:
            clstId = 0 if prjB[0] < prjB[1] else 1
            clstId = 2 if prjB[2] < prjB[clstId] else clstId
        elif prjB[0] < 0 and prjB[1] < 0 and prjB[2] < 0:
            clstId = 0 if prjB[0] > prjB[1] else 1
            clstId = 2 if prjB[2] > prjB[clstId] else clstId
        if clstId >= 0:
            ii = 0
            for ii in range(3):
                outter = np.cross(edgesA[ii], normalA)
                if float(np.dot(outter, nodesB[clstId] - nodesA[ii])) > 0:
                    break
            if ii == 3 or ii == 2 and False:
                clstPntB = nodesB[clstId]
                dist = float(prjB[clstId]) / (math.sqrt(lenNormalA2) if lenNormalA2 > 0 else 1.0)
                clstPntA = clstPntB - (normalA / (math.sqrt(lenNormalA2) if lenNormalA2 > 0 else 1.0)) * dist
                result.contactResult = ContactType.separate
                result.closetPtsPairs.append((clstPntA.copy(), clstPntB.copy()))
                return
    for i in range(3):
        edgeA = [nodesA[i], nodesA[(i + 1) % 3]]
        for j in range(3):
            edgeB = [nodesB[j], nodesB[(j + 1) % 3]]
            clA = np.zeros(3)
            clB = np.zeros(3)
            vec = np.zeros(3)
            SegmentDistance2(edgeA, edgeB, clA, clB, vec)
            if np.linalg.norm(clA - clB) < FLT_EPSILON:
                result.contPtsPairs.append((clA, clB))
    if len(result.contPtsPairs) < 1:
        result.contactResult = ContactType.separate
    else:
        result.contactResult = ContactType.surfaceToSurface
    return


def triTriIntersect(nodesA: List[np.ndarray], nodesB: List[np.ndarray], result: CollisionResult) -> None:
    A0, A1, A2 = nodesA
    B0, B1, B2 = nodesB
    E1 = A1 - A0
    E2 = A2 - A0
    Na = np.cross(E1, E2)
    d1 = -float(np.dot(Na, A0))
    db0 = float(np.dot(Na, B0) + d1)
    db1 = float(np.dot(Na, B1) + d1)
    db2 = float(np.dot(Na, B2) + d1)
    db0db1 = db0 * db1
    db0db2 = db0 * db2
    if db0db1 > FLT_EPSILON and db0db2 > FLT_EPSILON:
        result.contactResult = ContactType.separate
        return
    E1 = B1 - B0
    E2 = B2 - B0
    Nb = np.cross(E1, E2)
    d2 = -float(np.dot(Nb, B0))
    da0 = float(np.dot(Nb, A0) + d2)
    da1 = float(np.dot(Nb, A1) + d2)
    da2 = float(np.dot(Nb, A2) + d2)
    da0da1 = da0 * da1
    da0da2 = da0 * da2
    if da0da1 > FLT_EPSILON and da0da2 > FLT_EPSILON:
        result.contactResult = ContactType.separate
        return
    D = np.cross(Na, Nb)
    idx = int(np.argmax(np.abs(D)))
    ap0, ap1, ap2 = (A0[idx], A1[idx], A2[idx])
    bp0, bp1, bp2 = (B0[idx], B1[idx], B2[idx])
    coplanar, isectA0, isectA1, isectPointA1, isectPointA2 = computeIntervalsIsectline(A0, A1, A2, ap0, ap1, ap2, da0, da1, da2, da0da1, da0da2)
    if coplanar:
        coplanarTriTriIntersect(nodesA, nodesB, result)
        return
    _, isectB0, isectB1, isectPointB1, isectPointB2 = computeIntervalsIsectline(B0, B1, B2, bp0, bp1, bp2, db0, db1, db2, db0db1, db0db2)
    if isectA0 > isectA1:
        isectA0, isectA1 = (isectA1, isectA0)
        isectPointA1, isectPointA2 = (isectPointA2, isectPointA1)
    if isectB0 > isectB1:
        isectB0, isectB1 = (isectB1, isectB0)
        isectPointB1, isectPointB2 = (isectPointB2, isectPointB1)
    if isectA1 < isectB0 or isectB1 < isectA0:
        result.contactResult = ContactType.separate
        return
    computeIntersectionPoints([isectA0, isectA1], [isectB0, isectB1], isectPointA1, isectPointA2, isectPointB1, isectPointB2, result)
    if isectA0 > isectB0 and isectA1 < isectB1 or (isectB0 > isectA0 and isectB1 < isectA1):
        pairs = []
        N = None
        firstA = False
        if isectA0 > isectB0 and isectA1 < isectB1:
            pairs = [(da0, A0), (da1, A1), (da2, A2)]
            N = Nb
            firstA = True
        else:
            pairs = [(db0, B0), (db1, B1), (db2, B2)]
            N = Na
            firstA = False
        for di, Pi in pairs:
            if di > 0:
                continue
            clsP = Pi - di * N / float(np.dot(N, N))
            if firstA:
                res_inside = isInsideTriangle(clsP, [B0, B1, B2])
                result.contPtsPairs.append((Pi, clsP))
            else:
                res_inside = isInsideTriangle(clsP, [A0, A1, A2])
                result.contPtsPairs.append((clsP, Pi))
            if not res_inside and result.rejectPntsOutTri:
                result.contactResult = ContactType.unknownContact
                return
        if len(result.contPtsPairs) == 1:
            result.contactResult = ContactType.pointToSurface
        elif len(result.contPtsPairs) == 2:
            result.contactResult = ContactType.edgeToSurface
        else:
            result.contactResult = ContactType.unknownContact
            return
    elif isectB0 >= isectA0 and isectA1 <= isectB1 or (isectA0 >= isectB0 and isectB1 <= isectA1):
        if isectB0 >= isectA0 and isectA1 <= isectB1:
            isectPtB = result.isectpt1
            isectPtA = result.isectpt2
        else:
            isectPtB = result.isectpt2
            isectPtA = result.isectpt1
        clstA = None
        clstB = None
        defined = False
        for i in range(3):
            edgeA = [nodesA[i], nodesA[(i + 1) % 3]]
            for j in range(3):
                edgeB = [nodesB[i], nodesB[(i + 1) % 3]]
                vec = np.zeros(3)
                clA = np.zeros(3)
                clB = np.zeros(3)
                SegmentDistance2(edgeA, edgeB, clA, clB, vec)
                diff = clA - clB
                nrm = np.linalg.norm(diff)
                if nrm == 0:
                    continue
                vecAB = diff / nrm
                if np.dot(vecAB, Na) / (np.linalg.norm(Na) + 1e-18) < math.cos(result.peneAngleThres) or -np.dot(vecAB, Nb) / (np.linalg.norm(Nb) + 1e-18) < math.cos(result.peneAngleThres):
                    continue
                if not defined or (np.linalg.norm(clstA - clstB) > np.linalg.norm(diff) if defined else True):
                    clstA = clA
                    clstB = clB
                    defined = True
        if defined:
            result.contactResult = ContactType.edgeToEdge
            resA = isInsideTriangle(clstA, [A0, A1, A2])
            resB = isInsideTriangle(clstB, [B0, B1, B2])
            result.contPtsPairs.append((clstA, clstB))
            if not (resA and resB) and result.rejectPntsOutTri:
                result.contactResult = ContactType.unknownContact
                return
        else:
            result.contactResult = ContactType.unknownContact
            return
    else:
        result.contactResult = ContactType.unknownContact
        return
    n = result.contPtsPairs[0][1] - result.contPtsPairs[0][0]
    nn = np.linalg.norm(n)
    result.contactNormal = n / nn if nn > 0 else np.zeros(3)