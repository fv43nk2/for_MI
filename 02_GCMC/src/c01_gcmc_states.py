### --- import block --- ###
import numpy as np

### --- class block --- ###


# ==========================================================
# BaseState（共通責務）
# ==========================================================
class BaseState:
    """
    Abstract base class for GCMC states.

    Responsibilities:
    - Hold thermodynamic state (N, energy)
    - Record statistics
    - Define interface for Engine

    Does NOT:
    - Know Metropolis logic
    """

    def __init__(self):
        self.N_ads = 0
        self.total_energy = 0.0

        # statistics
        self.ads_history = []
        self.energy_history = []
        self.N_history = []

    # ------------------------------
    # observables
    # ------------------------------

    def get_total_energy(self):
        return self.total_energy

    # ------------------------------
    # statistics
    # ------------------------------

    def record_state(self):
        self.ads_history.append(self.get_loading())
        self.energy_history.append(self.total_energy)
        self.N_history.append(self.N_ads)

    def reset_statistics(self):
        self.ads_history = []
        self.energy_history = []
        self.N_history = []

    # ------------------------------
    # abstract interface
    # ------------------------------

    def get_loading(self):
        raise NotImplementedError

    def propose_insert(self):
        raise NotImplementedError

    def propose_delete(self):
        raise NotImplementedError

    def delta_energy_insert(self, trial):
        raise NotImplementedError

    def delta_energy_delete(self, trial):
        raise NotImplementedError

    def apply_insert(self, trial):
        raise NotImplementedError

    def apply_delete(self, trial):
        raise NotImplementedError


# ==========================================================
# LatticeState（格子ガス吸着モデル）
# ==========================================================


class LatticeState(BaseState):

    def __init__(self, energies: np.ndarray):
        super().__init__()

        self.energies = energies  # (N_sites,)
        self.N_sites = len(energies)

        self.occupancy = np.zeros(self.N_sites, dtype=np.int8)

    # ------------------------------
    # observables
    # ------------------------------

    def get_loading(self):
        return self.N_ads / self.N_sites

    # ------------------------------
    # proposal
    # ------------------------------

    def propose_insert(self):
        empty = np.where(self.occupancy == 0)[0]
        if len(empty) == 0:
            return None
        return np.random.choice(empty)

    def propose_delete(self):
        filled = np.where(self.occupancy == 1)[0]
        if len(filled) == 0:
            return None
        return np.random.choice(filled)

    # ------------------------------
    # energy
    # ------------------------------

    def delta_energy_insert(self, site):
        if site is None or self.occupancy[site] == 1:
            return None
        return float(self.energies[site])  # ← スカラー保証

    def delta_energy_delete(self, site):
        if site is None or self.occupancy[site] == 0:
            return None
        return float(-self.energies[site])

    # ------------------------------
    # apply
    # ------------------------------

    def apply_insert(self, site):
        self.occupancy[site] = 1
        self.N_ads += 1
        self.total_energy += self.energies[site]

    def apply_delete(self, site):
        self.occupancy[site] = 0
        self.N_ads -= 1
        self.total_energy -= self.energies[site]

    # --------------------------------
    # capability to insert/delete
    # -------------------------------

    def can_insert(self):
        return self.N_ads < self.N_sites

    def can_delete(self):
        return self.N_ads > 0

    # ------------------------------
    # utility
    # ------------------------------

    def copy(self):
        new = LatticeState(self.energies.copy())
        new.occupancy = self.occupancy.copy()
        new.N_ads = self.N_ads
        new.total_energy = self.total_energy
        return new


# ==========================================================
# ContinuousState（連続空間モデル)：RASPA-like
# ==========================================================
class ContinuousState(BaseState):
    """
    Continuous-space approximation using precomputed energy grid.

    Assumptions:
    - No adsorbate-adsorbate interaction
    - External field only (framework)

    設計メモ：
    ⇒ 吸着質同士の相互作用を考慮しない設計は破綻している（ので修正中）

    """

    def __init__(self, energy_grid: np.ndarray, box: np.ndarray, ff_params: dict):
        # 新たに ff_paramsを導入: RaspaForceFieldからの取得を想定していると思われるが、
        # 当該クラスの中身を知らないAIの出力なので、そのままは使えない（設計イメージだけと受け取るべし）
        super().__init__()
        self.energy_grid = energy_grid
        self.box = box  # [Lx, Ly, Lz] (メートル単位を想定)
        self.nx, self.ny, self.nz = energy_grid.shape
        self.dx = box[0] / self.nx
        self.dy = box[1] / self.ny
        self.dz = box[2] / self.nz

        # 分子座標のリスト (N_ads, 3)
        self.adsorbate_positions = []

        # 分子間相互作用パラメータ (CO2-CO2)
        # RaspaForceFieldを見に行けば分かるが、以下のような名称ではないので注意
        self.eps_kk = ff_params['eps_kk']  # [J] または [K] (単位系を統一)
        self.sig_kk = ff_params['sig_kk']  # [m]
        self.cutoff = 12.0e-10  # [m]
        self.cutoff2 = self.cutoff**2

    # def __init__(self, energy_grid: np.ndarray, box: np.ndarray):
    #     super().__init__()

    #     self.energy_grid = energy_grid
    #     self.box = box

    #     #
    #     self.nx, self.ny, self.nz = energy_grid.shape

    # ------------------------------
    # observables
    # ------------------------------

    def get_loading(self):
        # continuousでは占有率の定義が曖昧
        # とりあえず粒子数で返す（後で密度定義に拡張することを検討）
        return self.N_ads

    # ------------------------------
    # proposal
    # ------------------------------

    def propose_insert(self):

        # initialize positions at random
        pos = np.random.uniform([0, 0, 0], self.box)

        ix = int(pos[0] / self.box[0] * self.nx)
        iy = int(pos[1] / self.box[1] * self.ny)
        iz = int(pos[2] / self.box[2] * self.nz)

        # 境界安全化
        ix = min(ix, self.nx - 1)
        iy = min(iy, self.ny - 1)
        iz = min(iz, self.nz - 1)

        return (ix, iy, iz)

    def propose_delete(self):
        if self.N_ads == 0:
            return None

        # 相互作用なしモデルでは粒子識別不要
        return True

    # ------------------------------
    # energy
    # ------------------------------

    def delta_energy_insert(self, idx):
        if idx is None:
            return None

        ix, iy, iz = idx
        return float(self.energy_grid[ix, iy, iz])  # ← スカラー

    def delta_energy_delete(self, idx):
        # 本来は粒子位置が必要だが、
        # 相互作用なしモデルでは平均的に0扱いでも回る
        return 0.0

    # ------------------------------
    # apply
    # ------------------------------

    def apply_insert(self, idx):
        self.N_ads += 1
        self.total_energy += self.delta_energy_insert(idx)

    def apply_delete(self, idx):
        self.N_ads -= 1
        # エネルギー更新は簡略化（相互作用なし前提）

    # --------------------------------
    # capability to insert/delete
    # -------------------------------
    def can_insert(self):
        return True

    def can_delete(self):
        return self.N_ads > 0

    # ------------------------------
    # utility
    # ------------------------------

    def copy(self):
        new = ContinuousState(self.energy_grid.copy(), self.box.copy())
        new.N_ads = self.N_ads
        new.total_energy = self.total_energy
        return new


### --- 以下はContinuousState およびLatticeStateクラス完成後には不要になる旧State --- ###
# class GCMCState:
#     """
#     Represents the microscopic state of a GCMC system.

#     Responsibilities:
#     - Store occupancy configuration
#     - Track number of adsorbed molecules
#     - Track system energy
#     - Record statistics
#     - Apply accepted MC moves

#     No Metropolis logic inside.

#     設計メモ：
#         - 系のミクロ状態を管理
#             とはつまり、
#             状態を保持し、
#             外部Engineから与えられる指示――サイト占有率やエネルギーなどの変化――に対応し、
#             状態を更新すること
#             を意味する

#             あと、地味に履歴も記録する

#     """

#     def __init__(self, energies: np.ndarray):
#         """
#         Parameters
#         ----------
#         energies : np.ndarray
#             Site adsorption energies ε (J/mol)
#             ⇒ 将来的にはRASPAから与えられるようにする
#             ⇒ p-GCMC段階では、EnergyDistributorクラスからepsを受け取る仕組み

#         """

#         self.energies = energies  # これは外でAVOGADROを掛けて単位換算済み
#         self.N_sites = len(energies)

#         # debug
#         self.N_history = []  # 吸着サイト数の履歴を保存

#         # Occupancy array (0 or 1)
#         self.occupancy = np.zeros(self.N_sites, dtype=np.int8)

#         # Total adsorbed molecules
#         self.N_ads = 0

#         # Total energy
#         self.total_energy = 0.0

#         # Statistics
#         self.ads_history = []
#         self.energy_history = []

#     # ==========================================================
#     # Basic Observables
#     # ==========================================================

#     def get_occupancy_fraction(self):
#         return self.N_ads / self.N_sites

#     def get_total_energy(self):
#         return self.total_energy

#     # ==========================================================
#     # Move Proposals (energy difference only)
#     # ==========================================================

#     def delta_energy_insert(self, site_index):
#         if self.occupancy[site_index] == 1:
#             return None  # invalid move
#         return self.energies[site_index]

#     def delta_energy_delete(self, site_index):
#         if self.occupancy[site_index] == 0:
#             return None
#         return -self.energies[site_index]

#     # ==========================================================
#     # Apply Moves (only after acceptance)
#     # ==========================================================

#     def apply_insert(self, site_index):
#         self.occupancy[site_index] = 1
#         self.N_ads += 1
#         self.total_energy += self.energies[site_index]

#     def apply_delete(self, site_index):
#         self.occupancy[site_index] = 0
#         self.N_ads -= 1
#         self.total_energy -= self.energies[site_index]

#     # ==========================================================
#     # Sampling Utilities
#     # ==========================================================

#     def random_empty_site(self):
#         empty_sites = np.where(self.occupancy == 0)[0]
#         if len(empty_sites) == 0:
#             return None
#         return np.random.choice(empty_sites)

#     def random_filled_site(self):
#         filled_sites = np.where(self.occupancy == 1)[0]
#         if len(filled_sites) == 0:
#             return None
#         return np.random.choice(filled_sites)

#     # ==========================================================
#     # Statistics Recording
#     # ==========================================================

#     def record_state(self):
#         self.ads_history.append(self.N_ads / self.N_sites)
#         self.energy_history.append(self.total_energy)
#         self.N_history.append(self.N_ads)

#     def reset_statistics(self):
#         self.ads_history = []
#         self.energy_history = []
#         self.N_history = []

#     # ==========================================================
#     # Utility
#     # ==========================================================

#     def copy(self):
#         """
#         Deep copy of state (for replica exchange etc.)
#         """
#         new_state = GCMCState(self.energies.copy())
#         new_state.occupancy = self.occupancy.copy()
#         new_state.N_ads = self.N_ads
#         new_state.total_energy = self.total_energy

#         return new_state
