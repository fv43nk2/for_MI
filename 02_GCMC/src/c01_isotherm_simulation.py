### --- import block ---- ###
import numpy as np
from c01_chemical_potential_models import ChemicalPotentialModel
from typing import Optional, Union
from scipy.constants import h, k, Avogadro, R
from enum import Enum
from c01_gcmc_engines import GCMCEngine
from c01_gcmc_states import LatticeState, ContinuousState

# from c01_energy_calculators import EnergyCalculator


### --- class block ---- ###


# -------------------#
# ISotherm Simulator
# -------------------#
class IsothermSimulator:

    @staticmethod
    def run_isotherm(
        energy_calc,
        box,
        energies,
        dV,
        T: float,
        pressures: np.ndarray,  #
        m_molar: float,  # kg/mol
        mu0_value: Optional[float] = None,  # autoモードでは値を入力してあっても無意味になる
        mu0_mode: str = "auto",  # ["auto", "constant", "zero"] 連続空間モデル前提ならauto(旧コードとの整合性検証段階ならconstantでまず行うべきだが)
        adsorption_model: str = "continuous",  # 初期値は連続空間モデルとする(タイポの原因なのでEnumを検討すべき)
        n_steps: int = int(2e5),
        burn_in: int = 1000,
        record_interval: int = 100,
    ):
        """
        格子吸着モデル用 Isotherm Simulator

        前提：
        - エネルギーはサイトごとに固定
        - GCMCは(M-N)/(N+1) 重みを含む
        """

        # =========================================
        # エネルギー取得
        # =========================================
        all_energies = energy_calc.energy_grid  # [J/mol] energy grid from RASPA data

        mask = np.isfinite(all_energies)
        energies = all_energies[mask]

        #
        # print(f"[DEBUG][IsoSimulator] energy stats: start---")
        # print(f"mean: {np.mean(energies)}")
        # print(f"min/max: {np.min(energies)}, {np.max(energies)}")
        # print(f"[DEBUG][IsoSimulator] energy stats end ---")

        # =========================================
        # 化学ポテンシャルモデル
        # =========================================
        mu_model = ChemicalPotentialModel(
            m_molar, box, mu0_mode=mu0_mode, mu0_value=mu0_value, adsorption_model=adsorption_model
        )

        # m_molar = params["adsorbate"]["m_molar"]  # methan 0.016  # kg/mol

        mc_loadings = []
        N_history_all = []

        # =========================================
        # 圧力ループ
        # =========================================
        for P in pressures:

            mu = mu_model.compute_mu(T, P)  # ここでループの度に警告文が呼び出されるので困る

            # --- branched with adsorption model
            if adsorption_model == "lattice":
                state = LatticeState(energies)

            elif adsorption_model == "continuous":
                state = ContinuousState(energy_grid=all_energies, box=box)

            else:
                raise ValueError(f'GCMCstate has Unknown model: {adsorption_model}')

            engine = GCMCEngine(
                state=state, temperature=T, mu=mu, m_molar=m_molar, box=box, adsorption_model=adsorption_model
            )

            # 履歴初期化
            state.ads_history = []
            state.N_history = []

            engine.run(n_steps=n_steps, burn_in=burn_in, record_interval=record_interval)

            # 被覆率
            theta = np.mean(state.ads_history)
            mc_loadings.append(theta)

            # N分布保存
            N_history_all.append(state.N_history.copy())

        mc_isotherm = np.array(mc_loadings)

        # =========================================
        # 解析解
        # =========================================
        analytic_iso = IsothermSimulator._analytic_isotherm(energies, dV, pressures, T, mu_model, adsorption_model)

        valid_data = {"energies": energies}  # 何もvalidしてないけど？

        return mc_isotherm, analytic_iso, valid_data, mu, N_history_all

    # ==========================================================
    # analytic
    # ==========================================================
    @staticmethod
    def _analytic_isotherm(energies, dV, pressures, T, mu_model, adsorption_model):

        beta = 1 / T  # Unit: adjust to Raspa

        loadings = []

        for P in pressures:
            mu = mu_model.compute_mu(T, P)

            if adsorption_model == "lattice":
                # lattice model
                occ = 1.0 / (1.0 + np.exp(beta * (energies - mu)))
                theta = np.mean(occ)  #  theta = 1/N_site * \Sigma{occ}

            elif adsorption_model == "continuous":
                # continuous spacial model：accessible volume を考慮
                occ = np.exp(beta * (mu - energies))  # Boltzmann

                theta = np.sum(occ * dV)

            else:
                raise ValueError(f"Unknown adsorption_model: {adsorption_model}")

            loadings.append(theta)

        return np.array(loadings)

    # def _analytic_isotherm(energies, pressures, T, mu_model):

    #     beta = 1.0 / (R * T)

    #     loadings = []

    #     for P in pressures:

    #         mu = mu_model.compute_mu(T, P)
    #         print(f"[DEBUG][IsothermSimulator][analytic isotherm] mu: {mu}")

    #         occ = 1.0 / (1.0 + np.exp(beta * (energies - mu)))
    #         print(f"[DEBUG][IsothermSimulator][analytic isotherm] enegies/mu: {energies/mu}")

    #         theta = np.mean(occ)

    #         loadings.append(theta)

    #     return np.array(loadings)


# class IsothermSimulator:
#     """
#     設計メモ：
#     - Engine, mu_model(ChemicalPotentialModel)を外から渡す設計にすべき

#     - 将来的にはmu_sweepを実装すべき
#         具体的には、多成分系, 非理想気体, 相転移近傍, μ固定シミュレーション
#         を行う際に、直接muを制御する必要が生じる
#     """

#     def run_isotherm(params: dict, T: float, pressures: np.ndarray, mu0_value: float):
#         """
#         params: dict


#             Expected format:
#             {
#                 "grid_points": gridpoints,


#                 "framework":{
#                     "framework_name": framework_name,
#                     "framework_positions": framework_positions: np.ndarray,
#                     "framework_sigmas": framework_sigmas: np.ndarray, # これも配列
#                     "framework_epsilons": framework_epsilons: np.ndarray, # これも配列
#                 },

#                 "adsorbate":{
#                     "adsorbate_name": adsorbate_name: str,
#                     "adsorbate_sigma": ads_sigma: float, # こちらは配列ではない？？
#                     "adsorbate_epsilon": ads_epsilon: float,
#                 },

#                 "boundary":{
#                     "box": box: np.ndarray, # [Lx, Ly, Lz] 格子間距離とイコール
#                     "cutoff": cutoff : float,  # 手動設定（京大のリンク先を参照するに、もう少し賢い決め方がある）
#                 }
#             }

#         """

#         # デバッグ用
#         def analytic_isotherm(energies, pressures, T, mu_model):  # , m_molar

#             R = 8.314462618
#             beta = 1.0 / (R * T)
#             loadings = []

#             # mu_model = ChemicalPotentialModel(mu0_value=0.0)

#             # debug
#             print(f"energies[0]_analytic_isotherm:{energies[0]}")

#             for P in pressures:
#                 mu = mu_model.compute_mu(T, P)  # , m_molar

#                 occ = 1.0 / (1.0 + np.exp(beta * (energies - mu)))
#                 theta = np.mean(occ)
#                 loadings.append(theta)

#                 isotherm = np.array(loadings)

#             return isotherm

#         # extract params
#         grid_points = params["grid_points"]  # 吸着相の離散化を表現。frameworkに紐づいているわけではないため、別扱い
#         m_molar = params["adsorbate"]["m_molar"]  # methan 0.016  # kg/mol
#         box = params["boundary"]["box"]

#         # ここでparamsからEnergyCalculator用のパラメータだけを展開して渡す……と思ったが、面倒なのでやめ
#         e_calculator = EnergyCalculator(params)

#         AVOGADRO = 6.02214076e23
#         all_energies = e_calculator.compute_energy_grid(grid_points) * AVOGADRO  # J/mol換算

#         # デバッグ
#         # deb_energies = all_energies[np.isfinite(all_energies)]
#         # print(f'np.min(energies):{np.min(deb_energies)}, np.max(energies):{np.max(deb_energies)}')
#         # print(f'np.percentile(energies, [90, 95, 99, 99.9]): {np.percentile(deb_energies, [90, 95, 99, 99.9])}')

#         mask = all_energies < 1000  # 1e10
#         valid_data = {"coords": grid_points[mask], "energies": all_energies[mask]}

#         print(f"np.mean(valid_data['energies'])_run_isotherm:{np.mean(valid_data['energies'])}")

#         # 4. シミュレーションへ
#         state = GCMCState(valid_data["energies"])

#         mu_model = ChemicalPotentialModel(mu0_mode="constant", mu0_value=mu0_value)
#         # state = GCMCState(energies)

#         mc_loadings = []
#         # loadings = []

#         for P in pressures:

#             mu = mu_model.compute_mu(T, P)  # , m_molar
#             engine = GCMCEngine(state, T, mu, m_molar, box)  # Pはデバッグ用に一時的に入れただけ

#             # engine.run(n_steps=int(1e4), burn_in=1000, record_interval=10)  # equil　こちらは不要っぽい
#             state.ads_history = []

#             engine.run(
#                 n_steps=int(2e5), burn_in=1000, record_interval=100
#             )  # production  burn_in, record_intervalをむやみに増やすと結果が出力されなくなる

#             # debug N分布を見るために履歴保存
#             N_history = engine.state.N_history

#             theta = np.mean(state.ads_history)
#             mc_loadings.append(theta)

#         mc_isotherm = np.array(mc_loadings)

#         analytic_iso = analytic_isotherm(valid_data["energies"], pressures, T, mu_model)  # , m_molar

#         return mc_isotherm, analytic_iso, valid_data, mu, N_history
