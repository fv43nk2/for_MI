### --- import block --- ###
import numpy as np
import csv
import pandas as pd
from scipy.constants import h, k, Avogadro, R


### --- class block --- ###
class GCMCEngine:
    """
    Grand Canonical Monte Carlo Engine.

    Responsibilities:
    - Perform Metropolis MC moves
    - Maintain detailed balance
    - Interact with GCMCState
    - No knowledge of energy distribution structure

    Does NOT:
    - Store system state internally
    - Compute chemical potential
    """

    def __init__(self, state, temperature: float, mu: float, m_molar: float, box: np.ndarray, adsorption_model: str):
        """
        Parameters
        ----------
        state : GCMCState
        temperature : float (K)
        mu : float (J/mol)
        box: np.ndarray (m: meter)
        adsorption_model:str
        """

        self.state = state
        self.temperature = temperature  # [K] 系の浴温度
        self.mu = self._valid_mu(mu)

        self.R_GAS = R  # [J/mol/K] # モル当たりのEgを表す気体定数
        self.beta = 1.0 / (k * self.temperature)
        self.m_molar = m_molar  # 0.016  # kg/mol

        self.m_particle = (self.m_molar * 1e-3) / 6.02214076e23
        self.Lambda = h / np.sqrt(2 * np.pi * self.m_particle * k * self.temperature)
        self.adsorption_model = adsorption_model  # default "continious"
        self.V = box[0] * box[1] * box[2]  #  / (pow(angstrom, 3))

        # Statistics(とりあえずレジュームは考えていないので初期値は常に0でOK)
        self.n_attempt_insert = 0
        self.n_accept_insert = 0
        self.n_attempt_delete = 0
        self.n_accept_delete = 0

    # ==========================================================
    # Public Run Method
    # ==========================================================

    def run(self, n_steps: int, burn_in: int = 0, record_interval: int = 100, pressure_label: float = 0.0):
        self.state.reset_statistics()

        insert_history = {"step": [], "deltaE": [], "prob": []}  # , "eps_by_RT": []}
        delete_history = {"step": [], "deltaE": [], "prob": []}  # , "eps_by_RT": []}

        for step in range(n_steps):
            move_type = self._choose_move()

            if move_type == "insert":
                # _attempt_insertは (deltaE, prob, eps_by_RT) を返すと仮定
                deltaE_i, prob_i = self._attempt_insert()  # , e_rt_i
                # deltaE_d, prob_d, e_rt_d = self._attempt_delete()

                if step >= burn_in:  # 全件保存すると重いのでburn_in以降のみ推奨
                    insert_history["step"].append(step)
                    insert_history["deltaE"].append(deltaE_i)
                    insert_history["prob"].append(prob_i)
                    # insert_history["eps_by_RT"].append(e_rt_i)

                # debug

            else:
                deltaE_d, prob_d = self._attempt_delete()  # , e_rt_d

                if step >= burn_in:  # 全件保存すると重いのでburn_in以降のみ推奨
                    delete_history["step"].append(step)
                    delete_history["deltaE"].append(deltaE_d)
                    delete_history["prob"].append(prob_d)
                    # delete_history["eps_by_RT"].append(e_rt_d)

            if step >= burn_in and step % record_interval == 0:
                self.state.record_state()

    # ==========================================================
    # Move Selection
    # ==========================================================

    def _choose_move(self):
        """

        does not:
            - determine capability to insert/delete
        """
        can_ins = self.state.can_insert()
        can_del = self.state.can_delete()

        if not can_del:
            return "insert"

        if not can_ins:
            return "delete"

        return "insert" if np.random.rand() < 0.5 else "delete"

    # ==========================================================
    # Insert Move
    # ==========================================================

    def _attempt_insert(self):
        self.n_attempt_insert += 1

        trial = self.state.propose_insert()

        if trial is None:
            return None, 0.0

        # DEBUG: 一時的にdeltaE=0とおく（後で必ず直す！）
        deltaE = self.state.delta_energy_insert(trial)
        # deltaE = 0.0
        # print(f"[DEBUG][GCMCEngine._attempt_insert] deltaE: {deltaE}")

        N = self.state.N_ads  # 現在の（埋まっている）吸着サイト数

        ### ---  branched with adsroption model --- ###
        if self.adsorption_model == "lattice":
            M = self.state.N_sites  # 全吸着サイト数(for lattice mode only)
            # lattice gas model
            prob_ins = ((M - N) / (N + 1)) * np.exp(-self.beta * (-self.mu))

            # prob_ins = np.exp(-self.beta * (deltaE - self.mu)) # 元々の式()

        elif self.adsorption_model == "continuous":
            # 理論式のΔU(N→N+1) = U(N+1)-U(N)はdeltaEそのものだが、負値を返すようになっている
            # 敢えて理論式通りの形に戻して扱うなら、下記のようにすればすっきりする
            # また、deltaEはndarrayなので、単純にmuを差し引こうとするとエラーになる
            term_exp = np.exp(self.beta * (self.mu - deltaE))
            prob_ins = (self.V / ((self.Lambda**3) * (N + 1))) * term_exp
            # print(f"[DEBUG][GCMCEngine._attempt_insert] mu: {self.mu}")
            # print(
            #     f"[DEBUG][GCMCEngine._attempt_insert] self.beta * (self.mu - deltaE): {self.beta * (self.mu - deltaE)}"
            # )
            # print(f"[DEBUG][GCMCEngine_attempt_insert] prob_ins: {prob_ins}")

        else:
            raise ValueError(f"GCMCEngine has Unknown model: {self.adsorption_model}")

        prob = min(1.0, prob_ins)

        # determining insert probability
        if np.random.rand() < prob:
            self.state.apply_insert(trial)
            self.n_accept_insert += 1

        return deltaE, prob  # , eps_by_RT

    # ==========================================================
    # Delete Move
    # ==========================================================

    def _attempt_delete(self):
        self.n_attempt_delete += 1

        trial = self.state.propose_delete()

        if trial is None:
            return None, 0.0

        deltaE = self.state.delta_energy_delete(trial)

        N = self.state.N_ads  # 現在の（埋まっている）吸着サイト数

        ### ---  branched with adsroption model --- ###
        if self.adsorption_model == "lattice":
            M = self.state.N_sites
            # lattice gas model
            # expの中身： -beta * (ΔE_total - ΔN * mu)
            # 削除なので ΔN = -1。よって -beta * (deltaE - (-1)*mu) = -beta * (deltaE + mu)
            # しかし、deltaE自体がすでにマイナスを含んでいるので注意が必要。
            term_exp = np.exp(-self.beta * (deltaE + self.mu))
            # print(f"[DEBUG][GCMCEngine._attempt_delete] mu: {self.mu}")

            prob_del = (N / (M - N + 1)) * term_exp

        elif self.adsorption_model == "continuous":

            term_exp = np.exp(-self.beta * (self.mu + deltaE))
            prob_del = (N * (self.Lambda**3) / self.V) * term_exp

        else:
            raise ValueError(f"Unknown model: {self.adsorption_model}")

        prob = min(1.0, prob_del)

        if np.random.rand() < prob:
            self.state.apply_delete(trial)
            self.n_accept_delete += 1

        return deltaE, prob

    # ==========================================================
    # Diagnostics
    # ==========================================================

    def acceptance_ratios(self):

        insert_ratio = self.n_accept_insert / self.n_attempt_insert if self.n_attempt_insert > 0 else 0.0

        delete_ratio = self.n_accept_delete / self.n_attempt_delete if self.n_attempt_delete > 0 else 0.0

        return {"insert_acceptance": insert_ratio, "delete_acceptance": delete_ratio}

    def set_mu(self, mu_new):
        """
        Update chemical potential (for isotherm scan)
        """
        self.mu = mu_new

        # "C:\Users\y_7up\Documents\projects\mi_dev\04_mof\output\gcmc\test_insert.csv"

    # ==========================================================
    # snapshot
    # ==========================================================

    def save_snapshot(filename, adsorbate_pos, framework_atoms=None):
        """
        adsorbate_pos: list or np.ndarray (N, 3)
        framework_atoms: list of (pos, species) などのフレームワーク情報

        ある圧力における吸着状態を確認する

        設計メモ:
        吸着質(ガス)の位置情報 adsorbate_pos をどこから取得するのかが不明確
        ⇒ というか、そんな情報はどこにも保持していない疑い

        """
        with open(filename, "w") as f:
            # 全原子数 (吸着種 + フレームワーク)
            n_ads = len(adsorbate_pos)
            n_frm = len(framework_atoms) if framework_atoms else 0
            f.write(f"{n_ads + n_frm}\n")
            f.write("Atoms. Timestep: 0\n")

            # フレームワークの書き出し (固定)
            if framework_atoms:
                for pos, species in framework_atoms:
                    f.write(f"{species} {pos[0]*1e10} {pos[1]*1e10} {pos[2]*1e10}\n")

            # 吸着種の書き出し (変動)
            for pos in adsorbate_pos:
                # 単位をメートルからÅに戻して書き出す
                f.write(f"C {pos[0]*1e10} {pos[1]*1e10} {pos[2]*1e10}\n")

    # ==========================================================
    # validate parameters
    # ==========================================================

    def _valid_mu(self, mu):
        if isinstance(mu, np.ndarray):
            raise ValueError(f"{mu}: mu must be scalar, but array was given")

        else:
            valid_mu = mu

            return valid_mu
