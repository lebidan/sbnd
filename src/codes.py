import torch
from scipy.io import loadmat  # type: ignore[import-untyped]
from .utils import get_rank_zero_logger

log = get_rank_zero_logger(__name__)


class LinearCode:
    def __init__(self, mat_file: str) -> None:
        if not mat_file.endswith(".mat"):
            mat_file += ".mat"
        matlab_data = loadmat(mat_file, squeeze_me=True)
        self.G = torch.tensor(matlab_data["G"], dtype=torch.int8)
        self.H = torch.tensor(matlab_data["H"], dtype=torch.int8)
        self.Ht = self.H.T
        self.n, self.k = matlab_data["n"], matlab_data["k"]
        self.m = self.H.shape[0]
        assert (self.k, self.n) == self.G.shape
        self.rate = self.k * 1.0 / self.n
        self.dmin: int = matlab_data["dmin"] if "dmin" in matlab_data else 0
        self.name = matlab_data["name"] if "name" in matlab_data else "Linear"
        log.info(f"Instantiating a {self} code from file: {mat_file}")

    def __repr__(self) -> str:
        return (
            f"{self.name}({self.n},{self.k},{self.dmin})"
            if self.dmin != 0
            else f"{self.name}({self.n},{self.k})"
        )

    def encode(self, u: torch.Tensor) -> torch.Tensor:
        """Encode the binary input tensor u.
        Input shape: (bs, k)
        Output shape: (bs, n)"""
        return (u @ self.G).bitwise_and(1)

    def syndrome(self, z: torch.Tensor) -> torch.Tensor:
        """Compute the binary syndrome tensor s for the binary input tensor z.
        Input shape: (bs, n)
        Output shape: (bs, m)"""
        return (z @ self.Ht).bitwise_and(1)

    def soft_syndrome(self, z: torch.Tensor) -> torch.Tensor:
        """Compute the soft-output syndrome tensor s for the real-valued input tensor z.
        Calculation uses the min-sum approximation.
        Input shape: (bs, n)
        Output shape: (bs, m)"""
        zxH = z[:, None] * self.H[None]  # (bs, n-k, n)
        zxH[zxH.abs() < 1e-8] = (
            torch.inf
        )  # replace 0s with infs (inf does not affect min nor sign)
        return zxH.sign().prod(2) * zxH.abs().min(2)[0]


if __name__ == "__main__":
    pass
