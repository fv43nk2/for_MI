### ---  import block --- ###
import numpy as np
from scipy.constants import h, k, Avogadro, R
from typing import Optional, Union


### --- class block --- ###
class ChemicalPotentialModel:
    """
    Chemical potential model for Grand Canonical Monte Carlo.

    Responsibilities:
    - Convert (T, P) → μ
    - Keep unit consistency (J/mol)
    - No knowledge of MC engine or state
    """

    def __init__(
        self,
        m_molar: float,  # [g/mol]
        box: np.ndarray,  # [m: meter] for debug (to calculate V)
        mu0_mode: str = "auto",
        mu0_value: Optional[float] = None,
        P0: float = 1.0e5,
        adsorption_model="continuous",
    ):
        """
        Responsibilities:
        - generate chemical potential model.

        Parameters
        ----------
        mu0_mode : str
            "constant", "zero"

        mu0_value : float
            Standard chemical potential μ° (J/mol)

        P0 : float
            Reference pressure (Pa)

        設計メモ：
        まず3モードを用意
        - "constant" : μ°を定数で与える
        - "zero" : μ° = 0（基準化）
        - "tabulated" : 将来拡張用

        将来的には、
        - フガシティfを導入し、実在気体モデルにも対応できるようにするが、今は理想気体を仮定
            μ = μ^0 + RT*ln(f/P^0)

        """

        self.m_molar = m_molar
        self.box = box  # for debug (to calculate V)
        self.V = box[0] * box[1] * box[2]
        self.mu0_mode = mu0_mode
        self.mu0_value = mu0_value  # 初期値0.0では基準化になるのでモードをzeroにするのと同じ(比較用なのでそれで十分)
        self.P0 = P0  # 大気圧が前提
        self.adsorption_model = adsorption_model
        self._validate_mode()

    # ==========================================================
    # Public Interface
    # ==========================================================
    def compute_mu(self, T: float, P: float) -> float:
        """
        Compute chemical potential μ(T,P)

        Parameters
        ----------
        T : float
            Temperature (K)

        P : float
            Pressure (Pa)

        Returns
        -------
        mu : float
            Chemical potential (J/mol)
        """

        mu0_value = self._compute_mu0(T)
        # print(f'mu0_by_chemical_potential_models:{mu0}')

        if P <= 0.0:
            raise ValueError("Pressure must be positive")

        if self.adsorption_model == "lattice":
            mu = self._mu_for_lattice_gas_adsorption_model(mu0_value, T, P)

        elif self.adsorption_model == "continuous":
            mu = self._mu_for_continuous_space_model(mu0_value, T, P)

        # print(f'[DEBUG][ChemicalPotential.compute_mu] mu: {mu}')
        return mu

    # ==========================================================
    # private method block
    # ==========================================================

    def _compute_mu0(self, T: float) -> float:

        # print(f"[DEBUG][ChemicalPotential][_compute_mu0] self.mu0_mode: {self.mu0_mode}")

        if self.mu0_mode == "constant":

            return self.mu0_value

        elif self.mu0_mode == "zero":
            return 0.0

        elif self.mu0_mode == "auto":

            m_particle = (self.m_molar * 1e-3) / 6.02214076e23
            Lambda = h / np.sqrt(2 * np.pi * m_particle * k * T)

            # m = self.m_molar  # kg/particle
            # Lambda = h / np.sqrt(2 * np.pi * m * k * T)

            beta = 1.0 / (k * T)
            mu0 = (1 / beta) * np.log(beta * self.P0 * Lambda**3)

            # print(f'[DEBUG][ChemicalPotentialModel] self.V: {self.V}')
            # print(f'[DEBUG][ChemicalPotentialModel] self.V / Lambda**3: {self.V / Lambda**3}')
            # print(f'[DEBUG][ChemicalPotentialModel][_compute_mu0] mu0: {mu0}')

            self.Lambda = Lambda  # 保存したいなら

            # print(f"[DEBUG][ChemicalPotentialModel][_compute_mu0] Lambda: {Lambda} ")
            # print(f"[DEBUG][ChemicalPotentialModel][_compute_mu0] beta * P * Lambda**3: {beta * P * Lambda**3} ")

            return mu0

        else:
            raise ValueError(f"Unknown mu0_mode: {self.mu0_mode}")

    def _mu_for_lattice_gas_adsorption_model(self, mu0_value: float, T: float, P: float) -> float:

        beta = 1.0 / (k * T)

        # print(f"[DEBUG][ChemicalPotentialModel._mu_for_lattice_gas_adsorption_model][mu0]: mu0_value")

        mu = mu0_value + (1 / beta) * np.log(beta * P)
        # print(f"mu: {mu}")

        return mu

    def _mu_for_continuous_space_model(self, mu0_value: float, T: float, P: float) -> float:  # , m_molar: float
        """
        連続空間モデル用のmu計算
        - compute_muの中で呼び出される形式の方が合理的(計算ロジックだけ担わせる)
        - 今のところ格子モデルと中身は変わらない(Lambda計算を_compute_mu0の責任としたため)

        """
        beta = 1.0 / (k * T)

        mu = mu0_value + (1 / beta) * np.log(P / self.P0)
        # print(f"mu: {mu}")
        # \mu_B=\mu_{ideal}^0+k_BT\ln(\beta P\phi)\tag{17} # \phi　がfugacity
        # kB = R / Avogadro

        # debug
        # print(f"[DEBUG][ChemicalPoteintialModel][_mu_for_continuous_space_model] mu / RT = {mu / (R * T)}")

        return mu

    def _validate_mode(self):
        """インスタンス作成時に一度だけバリデーションを行う"""
        if self.adsorption_model == "continuous":
            suitable_mode = ["auto"]  # 後から追加する可能性を考慮
            if self.mu0_mode not in suitable_mode:
                print(f"[WARNING] {self.mu0_mode} is not suitable for continuous spatial model.")
