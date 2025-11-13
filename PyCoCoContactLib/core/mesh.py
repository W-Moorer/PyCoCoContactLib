from dataclasses import dataclass
import numpy as np


@dataclass
class Mesh:
    V: np.ndarray
    F: np.ndarray

    def center_of_mass(self) -> np.ndarray:
        return np.asarray(self.V, float).mean(axis=0)

    def to_internal(self):
        import ss_compare as _ss
        return _ss.Mesh(np.asarray(self.V, float), np.asarray(self.F, int))

