import numpy as np
from enum import IntEnum
from dataclasses import dataclass, field
from typing import List, Tuple
import math


FLT_EPSILON = np.finfo(np.float32).eps
EILON_1 = 1e-6
EILON_2 = 1e-10


class ContactType(IntEnum):
    SEPARATE = 0
    UNKNOWN = 1
    INTERSECT_2D = 2
    OVERLAP_2D = 3
    POINT_TO_SURFACE = 4
    EDGE_TO_SURFACE = 5
    EDGE_TO_EDGE = 6
    SURFACE_TO_SURFACE = 7
    SPHERE_TO_SPHERE = 8
    SPHERE_TO_PLANE = 9
    SPHERE_TO_CUBOID = 10
    SPHERE_TO_CYLINDER = 11
    CUBOID_TO_CUBOID = 12
    CYLINDER_TO_PLANE = 13
    RBF_TO_RBF = 14
    TRIANGLE_TO_TRIANGLE = 15
    TRIANGLE_TO_SPHERE = 16
    TRIANGLE_TO_PLANE = 17


@dataclass
class CollisionResult:
    contPtsPair2D: Tuple[np.ndarray, np.ndarray] = (np.zeros(2), np.zeros(2))
    contactNormal2D: np.ndarray = field(default_factory=lambda: np.zeros(2))
    ids: Tuple[int, int] = (0, 0)
    isectpt1: np.ndarray = field(default_factory=lambda: np.zeros(3))
    isectpt2: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rejectPntsOutTri: bool = True
    peneAngleThres: float = math.pi / 4.0
    surfaceSurfaceAngleThres: float = 9.0 * math.pi / 10.0
    contactResult: ContactType = ContactType.UNKNOWN
    contPtsPairs: List[Tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    closetPtsPairs: List[Tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    contactNormal: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def setResult(self, cp1: np.ndarray, cp2: np.ndarray, ctype: ContactType) -> None:
        self.contactResult = ctype
        n = cp2 - cp1
        norm = np.linalg.norm(n)
        self.contactNormal = (n / norm) if norm > 0 else np.zeros(3)
        self.contPtsPairs.append((cp1, cp2))


@dataclass
class ContactPoint:
    triA: int
    triB: int
    baryA: np.ndarray
    baryB: np.ndarray
    pA_world: np.ndarray
    pB_world: np.ndarray
    n: np.ndarray
    phi: float
    area_weight: float = 1.0
    lifetime: int = 0


@dataclass
class ContactManifold:
    points: List[ContactPoint] = field(default_factory=list)

    def is_empty(self) -> bool:
        return len(self.points) == 0