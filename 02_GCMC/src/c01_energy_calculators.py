### --- import block --- ###
import numpy as np
from scipy.constants import Avogadro, R, k


### --- class block --- ###
class EnergyMatrixCalculator:

    def __init__(self, structure, force_field, grid, cutoff):

        self.structure = structure
        self.ff = force_field

        # すでに structure.positions は [m] 単位
        self.framework_positions = structure.positions
        self.framework_types = structure.framework_types

        # --- グリッド ---
        self.x = np.array(grid["x"])
        self.y = np.array(grid["y"])
        self.z = np.array(grid["z"])

        # グリッドも [m] 単位で受け取る（1e10倍しない）
        self.x = np.array(grid["x"])
        self.y = np.array(grid["y"])
        self.z = np.array(grid["z"])

        # generate dx, dy, dz
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]
        self.dz = self.z[1] - self.z[0]

        # --- box長（ここで定義する） ---
        self.Lx = self.x[-1] - self.x[0] + self.dx
        self.Ly = self.y[-1] - self.y[0] + self.dy
        self.Lz = self.z[-1] - self.z[0] + self.dz

        # --- grid ---
        self.xx, self.yy, self.zz = np.meshgrid(self.x, self.y, self.z, indexing="ij")

        self.cutoff2 = cutoff**2

        self.energy_grid = None

    # -----------------------------
    # PBC（minimum image）
    # -----------------------------
    def _minimum_image(self, d, L):
        return d - L * np.round(d / L)

    # -----------------------------
    # LJ field
    # -----------------------------
    # def _lj_field(self, atom_pos, eps_K, sig_m):
    #     # デバッグ用
    #     # 距離計算（m単位）
    #     dx = self._minimum_image(self.xx - atom_pos[0], self.Lx)
    #     dy = self._minimum_image(self.yy - atom_pos[1], self.Ly)
    #     dz = self._minimum_image(self.zz - atom_pos[2], self.Lz)
    #     r2 = dx * dx + dy * dy + dz * dz
    #     # 強制的に「吸着が起きるはずのエネルギー」を全グリッドにセットしてみる
    #     # -200K 相当のエネルギー [J]
    #     U = np.full_like(r2, -200.0)

    #     return U

    def _lj_field(self, atom_pos, eps_K, sig_m):
        # 距離計算（m単位）
        dx = self._minimum_image(self.xx - atom_pos[0], self.Lx)
        dy = self._minimum_image(self.yy - atom_pos[1], self.Ly)
        dz = self._minimum_image(self.zz - atom_pos[2], self.Lz)
        r2 = dx * dx + dy * dy + dz * dz

        # 1. 全域を非常に高い壁 [K] で初期化（初期値 0 ではなく 1e5）
        # ただし、後で足し合わせることを考えると 0 初期化で mask 外を 1e5 にする方が安全
        U = np.zeros_like(r2)

        # 2. 距離の閾値
        r_min2 = (0.8 * sig_m) ** 2

        # 3. マスクの考え方を変更：
        # 「計算する場所」は「カットオフ内」のすべて
        calc_mask = r2 < self.cutoff2
        # 「壁にする場所」は「近すぎる場所」
        overlap_mask = r2 <= r_min2

        if np.any(calc_mask):
            # メートル単位でLJ計算
            inv_r2 = (sig_m**2) / r2[calc_mask]
            inv_r6 = inv_r2**3
            inv_r12 = inv_r6**2

            # K 単位で計算
            U[calc_mask] = 4.0 * eps_K * (inv_r12 - inv_r6)

        # 4. 重なっている場所を巨大な斥力壁で上書き
        U[overlap_mask] = 1e5  # 10万ケルビンの壁

        # print(f"[DEBUG] r2 min/max: {np.min(r2):.2e} / {np.max(r2):.2e}")
        # print(f"[DEBUG] cutoff2: {self.cutoff2:.2e}")
        # print(f"[DEBUG] r_min2: {(0.8 * sig_m)**2:.2e}")
        # print(f"[DEBUG] active mask counts: {np.sum(calc_mask)} / {calc_mask.size}")
        # print(f"[DEBUG] U stats: min={np.min(U):.2e}, max={np.max(U):.2e}, mean={np.mean(U):.2e}")

        return U

    # -----------------------------
    # grid構築
    # -----------------------------
    def build_energy_grid(self):

        nx, ny, nz = self.xx.shape
        grid = np.zeros((nx, ny, nz))

        ads_type = self.ff.adsorbate_type_ids[0]

        for pos, frm_type in zip(
            self.framework_positions,
            self.framework_types,
        ):
            eps = self.ff.eps_matrix[ads_type, frm_type]  # K
            sig = self.ff.sig_matrix[ads_type, frm_type]  # Å

            # ここでメートルに変換して _lj_field に渡す
            sig_m = sig * 1e-10

            # print(f"k value: {k}")
            grid += self._lj_field(pos, eps, sig_m)  # * k  # return array of U

        # self.energy_grid = grid
        self.energy_grid = grid * k  # 1.380649e-23

        # print("--- Final Build Check ---")
        # print(f"grid_K min/max: {np.min(grid):.2e} / {np.max(grid):.2e}")
        # print(f"final energy_grid min/max: {np.min(self.energy_grid):.2e} / {np.max(self.energy_grid):.2e}")

    # -----------------------------
    # 補間
    # -----------------------------
    def _interpolate(self, pos):
        """
        三線形補間 (Trilinear Interpolation)
        - 周期境界条件 (PBC) を考慮し、端点でも安全に補間を行う。
        """
        nx, ny, nz = self.energy_grid.shape
        U = self.energy_grid

        # 1. 座標を 0 ~ L に正規化（グリッド開始点 self.x[0] からの相対距離）
        # % self.Lx により、Boxの外に提案された座標も自動的にBox内へ引き戻される
        rx = (pos[0] - self.x[0]) % self.Lx
        ry = (pos[1] - self.y[0]) % self.Ly
        rz = (pos[2] - self.z[0]) % self.Lz

        # 2. インデックスの算出
        i = int(rx / self.dx) % nx
        j = int(ry / self.dy) % ny
        k = int(rz / self.dz) % nz

        # 隣接インデックス（周期境界により末尾の次は0番目へ）
        ip1 = (i + 1) % nx
        jp1 = (j + 1) % ny
        kp1 = (k + 1) % nz

        # 3. 各軸内での相対位置 (0.0 ~ 1.0)
        # グリッド点 self.x[i] からの距離を dx で割る
        fx = (rx - (i * self.dx)) / self.dx
        fy = (ry - (j * self.dy)) / self.dy
        fz = (rz - (k * self.dz)) / self.dz

        # 4. 8頂点のエネルギー値を取得
        c000 = U[i, j, k]
        c100 = U[ip1, j, k]
        c010 = U[i, jp1, k]
        c110 = U[ip1, jp1, k]
        c001 = U[i, j, kp1]
        c101 = U[ip1, j, kp1]
        c011 = U[i, jp1, kp1]
        c111 = U[ip1, jp1, kp1]

        # 5. 三線形補間の実行
        # x方向の補間
        c00 = c000 * (1 - fx) + c100 * fx
        c10 = c010 * (1 - fx) + c110 * fx
        c01 = c001 * (1 - fx) + c101 * fx
        c11 = c011 * (1 - fx) + c111 * fx

        # y方向の補間
        c0 = c00 * (1 - fy) + c10 * fy
        c1 = c01 * (1 - fy) + c11 * fy

        # z方向の補間
        return c0 * (1 - fz) + c1 * fz

    def __call__(self, position):

        if self.energy_grid is None:
            raise RuntimeError("energy grid not built")

        return self._interpolate(position)

    def get_energy_distribution(self):

        if self.energy_grid is None:
            raise RuntimeError("energy grid not built")

        energies = self.energy_grid.flatten()
        dV = self.dx * self.dy * self.dz

        return energies, dV

    # def _lj_field(self, atom_pos, eps, sig):

    #     dx = self.xx - atom_pos[0]
    #     dy = self.yy - atom_pos[1]
    #     dz = self.zz - atom_pos[2]

    #     # PBC適用（最重要）
    #     dx = self._minimum_image(dx, self.Lx)
    #     dy = self._minimum_image(dy, self.Ly)
    #     dz = self._minimum_image(dz, self.Lz)

    #     r2 = dx * dx + dy * dy + dz * dz
    #     r = np.sqrt(r2)

    #     # r の下限をクリップ
    #     r = np.maximum(r, 1e-6)  # Åスケールで十分小さい値

    #     # debug
    #     # print("[DEBUG][_lj_field] grid stats")
    #     # print("xx min/max:", np.min(self.xx), np.max(self.xx))
    #     # print("yy min/max:", np.min(self.yy), np.max(self.yy))
    #     # print("zz min/max:", np.min(self.zz), np.max(self.zz))

    #     # print("[DEBUG][_lj_field] atom pos:", atom_pos)

    #     # print("[DEBUG][raw r stats]")
    #     # print("min:", np.min(r))
    #     # print("mean:", np.mean(r))
    #     # print("max:", np.max(r))

    #     # print("count raw r < 1.0:", np.sum(r < 1.0))
    #     # print("count raw r < 0.5:", np.sum(r < 0.5))
    #     # print("count raw r < 0.1:", np.sum(r < 0.1))
    #     # print("count raw r == 0:", np.sum(r == 0))

    #     # ratio = sig / r
    #     # print("[DEBUG][sigma/raw r]")
    #     # print("max:", np.max(ratio))

    #     # --- mask ---
    #     r_min = 0.8 * sig
    #     r_min2 = r_min * r_min

    #     mask = (r2 < self.cutoff2) & (r2 > r_min2)

    #     # --- 出力（グリッド形状を維持） ---
    #     U = np.full_like(r2, 1e10)

    #     if not np.any(mask):
    #         return U

    #     r2_valid = r2[mask]
    #     r_valid = np.sqrt(r2_valid)

    #     # debug guarded
    #     if np.any(mask):
    #         pass
    #         # print("[DEBUG][r_valid stats]")
    #         # print("min:", np.min(r_valid))
    #         # print("mean:", np.mean(r_valid))
    #         # print("max:", np.max(r_valid))
    #     else:
    #         print("[DEBUG] r_valid is empty")

    #     # print("count r_valid < 1.0:", np.sum(r_valid < 1.0))
    #     # print("count r_valid < 0.5:", np.sum(r_valid < 0.5))
    #     # print("count r_valid < 0.1:", np.sum(r_valid < 0.1))
    #     # print("count r_valid == 0:", np.sum(r_valid == 0))

    #     ratio_valid = sig / r_valid
    #     # print("[DEBUG][sigma/r_valid]")
    #     # print("max:", np.max(ratio_valid))

    #     print()

    #     inv_r2 = (sig * sig) / r2_valid
    #     inv_r6 = inv_r2**3
    #     inv_r12 = inv_r6**2
    #     # print("[DEBUG][LJ][with r_vlaid]")
    #     # print("inv_r6 max:", np.max(inv_r6))
    #     # print("inv_r12 max:", np.max(inv_r12))

    #     U = np.zeros_like(r2)

    #     U[mask] = 4.0 * eps * (inv_r12 - inv_r6)

    #     U_valid = U[mask]

    #     # --- 統計 ---
    #     # print(
    #     #     f"[DEBUg][EnergyMatrixCalculator._ij_field] U_valid stats:, min: {np.min(U_valid)}, mean: {np.mean(U_valid)}, max: {np.max(U_valid)}"
    #     # )

    #     # print()

    #     # print("mask true ratio:", np.sum(mask) / mask.size)
    #     # print("min r (all):", np.sqrt(np.min(r2)))
    #     # print("min r (valid):", np.min(r_valid) if len(r_valid) > 0 else None)

    #     return U

    # def _interpolate(self, pos):

    #     x, y, z = pos

    #     i = int((x - self.x[0]) / self.dx)
    #     j = int((y - self.y[0]) / self.dy)
    #     k = int((z - self.z[0]) / self.dz)

    #     fx = (x - self.x[i]) / self.dx
    #     fy = (y - self.y[j]) / self.dy
    #     fz = (z - self.z[k]) / self.dz

    #     U = self.energy_grid

    #     c000 = U[i, j, k]
    #     c100 = U[i + 1, j, k]
    #     c010 = U[i, j + 1, k]
    #     c110 = U[i + 1, j + 1, k]
    #     c001 = U[i, j, k + 1]
    #     c101 = U[i + 1, j, k + 1]
    #     c011 = U[i, j + 1, k + 1]
    #     c111 = U[i + 1, j + 1, k + 1]

    #     c00 = c000 * (1 - fx) + c100 * fx
    #     c10 = c010 * (1 - fx) + c110 * fx
    #     c01 = c001 * (1 - fx) + c101 * fx
    #     c11 = c011 * (1 - fx) + c111 * fx

    #     c0 = c00 * (1 - fy) + c10 * fy
    #     c1 = c01 * (1 - fy) + c11 * fy

    #     return c0 * (1 - fz) + c1 * fz


# class EnergyMatrixCalculator:
#     """
#     Responsibilities
#     ----------------
#     - frameworkとのLJポテンシャルから energy grid を構築
#     - 任意座標のポテンシャルを補間して返す

#     依存
#     ----
#     structure
#         - framework_positions (N,3)
#         - framework_types (N,)

#     force_field : RaspaForceField
#         - eps_matrix
#         - sig_matrix
#         - adsorbate_type_ids

#     grid
#         - x,y,z grid coordinate arrays
#     """

#     def __init__(self, structure, force_field, grid, cutoff):

#         self.structure = structure
#         self.ff = force_field
#         self.grid = grid
#         self.cutoff = cutoff

#         # framework
#         self.framework_positions = structure.positions  # ["framework_positions"]
#         self.framework_types = structure.framework_types  # ["framework_types"]

#         # grid
#         self.x = grid["x"]
#         self.y = grid["y"]
#         self.z = grid["z"]

#         # --- 単位統一: m → Å ---
#         self.framework_positions = self.framework_positions * 1e10

#         self.x = self.x * 1e10
#         self.y = self.y * 1e10
#         self.z = self.z * 1e10

#         # dx, dy, dz
#         self.dx = self.x[1] - self.x[0]
#         self.dy = self.y[1] - self.y[0]
#         self.dz = self.z[1] - self.z[0]

#         self.xx, self.yy, self.zz = np.meshgrid(self.x, self.y, self.z, indexing="ij")

#         self.energy_grid = None

#     # -------------------------------------------------
#     # Private
#     # -------------------------------------------------

#     def _lj_field(self, atom_pos, eps, sig):
#         """
#         1つのframework原子によるポテンシャル場
#         """

#         print("[DEBUG][_lj_field] grid stats")
#         print("xx min/max:", np.min(self.xx), np.max(self.xx))
#         print("yy min/max:", np.min(self.yy), np.max(self.yy))
#         print("zz min/max:", np.min(self.zz), np.max(self.zz))

#         print("[DEBUG][_lj_field] atom pos:", atom_pos)

#         dx = self.xx - atom_pos[0]
#         dy = self.yy - atom_pos[1]
#         dz = self.zz - atom_pos[2]

#         r2 = dx * dx + dy * dy + dz * dz
#         r = np.sqrt(r2)

#         print("[DEBUG][raw r stats]")
#         print("min:", np.min(r))
#         print("mean:", np.mean(r))
#         print("max:", np.max(r))

#         print("count raw r < 1.0:", np.sum(r < 1.0))
#         print("count raw r < 0.5:", np.sum(r < 0.5))
#         print("count raw r < 0.1:", np.sum(r < 0.1))
#         print("count raw r == 0:", np.sum(r == 0))

#         ratio = sig / r
#         print("[DEBUG][sigma/raw r]")
#         print("max:", np.max(ratio))

#         print()

#         cutoff2 = self.cutoff**2
#         mask = r2 < cutoff2

#         U = np.zeros_like(r2)

#         mask = (r2 > 0) & (r2 < cutoff2)

#         # 周期境界条件の処理
#         if not np.any(mask):
#             return np.zeros_like(r2)

#         r2_valid = r2[mask]
#         r_valid = np.sqrt(r2_valid)

#         # debug guarded
#         if np.any(mask):
#             print("min:", np.min(r_valid))
#             print("[DEBUG][r_valid stats]")
#             print("min:", np.min(r_valid))
#             print("mean:", np.mean(r_valid))
#             print("max:", np.max(r_valid))
#         else:
#             print("[DEBUG] r_valid is empty")

#         print("count r_valid < 1.0:", np.sum(r_valid < 1.0))
#         print("count r_valid < 0.5:", np.sum(r_valid < 0.5))
#         print("count r_valid < 0.1:", np.sum(r_valid < 0.1))
#         print("count r_valid == 0:", np.sum(r_valid == 0))

#         ratio_valid = sig / r_valid
#         print("[DEBUG][sigma/r_valid]")
#         print("max:", np.max(ratio_valid))

#         print()

#         inv_r2 = (sig * sig) / r2_valid
#         inv_r6 = inv_r2**3
#         inv_r12 = inv_r6**2

#         print("[DEBUG][LJ][with r_vlaid]")
#         print("inv_r6 max:", np.max(inv_r6))
#         print("inv_r12 max:", np.max(inv_r12))

#         U[mask] = 4 * eps * (inv_r12 - inv_r6)
#         print("[DEBUG][U contribution][with r_vlaid]")
#         print("min:", np.min(U))
#         print("max:", np.max(U))
#         print("mean:", np.mean(U))

#         return U

#     # -------------------------------------------------
#     # Energy grid construction
#     # -------------------------------------------------

#     def build_energy_grid(self):
#         """
#         framework全原子の寄与を足して energy grid を作る
#         """

#         nx, ny, nz = self.xx.shape

#         grid = np.zeros((nx, ny, nz))

#         # 単一吸着質を仮定
#         ads_type = self.ff.adsorbate_type_ids[0]

#         for pos, frm_type in zip(
#             self.framework_positions,
#             self.framework_types,
#         ):

#             eps = self.ff.eps_matrix[ads_type, frm_type]
#             # print(f"[DEBUG][EnergyMatrixCalculator] eps (no Unit Conversion): {eps}")

#             sig = self.ff.sig_matrix[ads_type, frm_type]

#             grid += self._lj_field(pos, eps, sig)

#         self.energy_grid = grid

#     # -------------------------------------------------
#     # Interpolation
#     # -------------------------------------------------

#     def _interpolate(self, pos):

#         x, y, z = pos

#         i = int((x - self.x[0]) / self.dx)
#         j = int((y - self.y[0]) / self.dy)
#         k = int((z - self.z[0]) / self.dz)

#         fx = (x - self.x[i]) / self.dx
#         fy = (y - self.y[j]) / self.dy
#         fz = (z - self.z[k]) / self.dz

#         U = self.energy_grid

#         c000 = U[i, j, k]
#         c100 = U[i + 1, j, k]
#         c010 = U[i, j + 1, k]
#         c110 = U[i + 1, j + 1, k]
#         c001 = U[i, j, k + 1]
#         c101 = U[i + 1, j, k + 1]
#         c011 = U[i, j + 1, k + 1]
#         c111 = U[i + 1, j + 1, k + 1]

#         c00 = c000 * (1 - fx) + c100 * fx
#         c10 = c010 * (1 - fx) + c110 * fx
#         c01 = c001 * (1 - fx) + c101 * fx
#         c11 = c011 * (1 - fx) + c111 * fx

#         c0 = c00 * (1 - fy) + c10 * fy
#         c1 = c01 * (1 - fy) + c11 * fy

#         return c0 * (1 - fz) + c1 * fz

#     # -------------------------------------------------
#     # Public API
#     # -------------------------------------------------

#     def __call__(self, position):
#         """
#         任意座標のエネルギーを返す
#         """

#         if self.energy_grid is None:
#             raise RuntimeError("energy grid not built")

#         return self._interpolate(position)

#     # とりあえずの対処(要検証)
#     def get_energy_distribution(self):
#         if self.energy_grid is None:
#             raise RuntimeError("energy grid not built")

#         energies = self.energy_grid.flatten()

#         dV = self.dx * self.dy * self.dz

#         return energies, dV

#     ### --- private method block --- ###
#     def _validate_params(self, params):
#         # paramsはそもそもRaspaForceFieldインスタンスから受け渡されるため、最低限のvalidationは実施済み

#         pass


# class EnergyCalculator:
#     """
#     設計メモ：現設計(格子点吸着モデルであり、行列計算にも非対応)
#     ⇒ 連続空間モデルかつ行列計算対応のクラスが完成したら封印

#     Responsibilities:
#     - calculate energy distribution with LJ potential.

#     Ref:
#     https://www.fml.t.u-tokyo.ac.jp/~izumi/CMS/MD/models.pdf

#     """

#     def __init__(
#         self,
#         params: dict,
#     ):
#         """
#         params: dict
#             Energy distribution settings.

#             Expected format:
#             {
#                 "framework":{
#                     "framework_name": framework_name,
#                     "framework_positions": framework_positions: np.ndarray,
#                     "framework_sigmas": framework_sigmas: np.ndarray, # これも配列
#                     "framework_epsilons": framework_epsilons: np.ndarray, # これも配列
#                 },

#                 "adsorbate":{
#                     "adsorbate_name": adsorbate_name: str,
#                     "adsorbate_sigma": ads_sigma: float, # 単原子分子ならfloatでOK。将来的に多原子にするなら配列。
#                     "adsorbate_epsilon": ads_epsilon: float,
#                 },

#                 "boundary":{
#                     "box": box: np.ndarray, # [Lx, Ly, Lz] 格子間距離とイコール
#                     "cutoff": cutoff : float,  # 手動設定（京大のリンク先を参照するに、もう少し賢い決め方がある）
#                 }
#             }

#         """

#         self.params = params
#         self._validate_params()

#         # --- 混合則を事前に計算しておく ---
#         self._precompute_mixing_parameters()

#     def _precompute_mixing_parameters(self):
#         """全ての骨格原子との混合LJパラメータを事前に配列として保持する"""
#         f_params = self.params["framework"]
#         ads_params = self.params["adsorbate"]

#         # 配列として一括計算 (Lorentz-Berthelot)
#         self.sigmas_ij = 0.5 * (f_params["framework_sigmas"] + ads_params["adsorbate_sigma"])
#         self.epsilons_ij = np.sqrt(f_params["framework_epsilons"] * ads_params["adsorbate_epsilon"])

#         # 重なり判定用の閾値 (σの0.8倍など) も事前に計算
#         sigma_eff = self.sigmas_ij
#         self.repulsive_wall = 0.9 * sigma_eff
#         # print(self.repulsive_wall)

#     def compute_site_energy(self, position):
#         """ベクトル化された1点ポテンシャル計算"""
#         boundary_params = self.params["boundary"]
#         box = boundary_params["box"]
#         cutoff2 = boundary_params["cutoff"] ** 2

#         # 1. 全骨格原子との相対距離を一括計算 (Shape: [N_atoms, 3])
#         dr_raw = position - self.params["framework"]["framework_positions"]

#         # 2. 最小画像則を一括適用
#         dr = dr_raw - box * np.round(dr_raw / box)

#         # 3. 距離の2乗を一括計算 (Shape: [N_atoms])
#         r2 = np.sum(dr**2, axis=1)

#         # --- 改善2: 判定の整理 ---

#         # A. 重なり判定 (近すぎる原子が1つでもあるか)
#         # 1つでも閾値を下回れば、そのサイトは「吸着不可」として巨大な値を返す
#         if np.any(r2 < self.repulsive_wall**2):
#             return np.inf

#         # B. カットオフ判定
#         # カットオフ内のインデックスだけを抽出
#         mask = r2 < cutoff2
#         if not np.any(mask):
#             return 0.0

#         # 有効な点のみ抽出して計算
#         valid_r2 = r2[mask]
#         valid_sig2 = self.sigmas_ij[mask] ** 2
#         valid_eps = self.epsilons_ij[mask]

#         # LJ計算 (ベクトル計算)
#         inv_r2 = valid_sig2 / valid_r2
#         inv_r6 = inv_r2**3
#         inv_r12 = inv_r6**2

#         energy = np.sum(4 * valid_eps * (inv_r12 - inv_r6))

#         return energy

#     # --- 格子点全体を事前計算 ---
#     def compute_energy_grid(self, grid_points):
#         """
#         本クラスの出力(エネルギー分布)を担当

#         本クラスでは格子吸着近似を仮定。
#         なお、RASPAは連続空間GCMC。
#         """
#         energies = []
#         for pos in grid_points:
#             e = self.compute_site_energy(pos)
#             energies.append(e)

#         epsilon_distribution = np.array(energies)

#         return epsilon_distribution

#     def _validate_params(self):
#         """
#         Minimal validation for robustness
#         """

#         pass
