from abc import ABC, abstractmethod
from .calculator import HertzLikeParameters, ContactForceCalculator


class ContactModel(ABC):
    @abstractmethod
    def compute_force(self, delta: float, delta_dot: float = 0.0) -> float:
        raise NotImplementedError


class HertzModel(ContactModel):
    def __init__(self, k: float = 1e6, n: float = 1.5, d: float = 1e3):
        self.params = HertzLikeParameters(k=k, n=n, d=d)

    def compute_force(self, delta: float, delta_dot: float = 0.0) -> float:
        calc = ContactForceCalculator(self.params)
        return calc.compute_point_force_magnitude(delta, delta_dot)
