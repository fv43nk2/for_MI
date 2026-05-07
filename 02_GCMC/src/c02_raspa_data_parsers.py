### --- import block --- ###
import json
import numpy as np

### --- class block --- ###
import json
import numpy as np


class RaspaForceField:

    def __init__(self, json_path):

        with open(json_path, "r") as f:
            self.data = json.load(f)

        # -----------------------------
        # global params
        # -----------------------------
        self.mixing_rule = self.data.get("MixingRule", "Lorentz-Berthelot")

        # ★ cutoffはÅで強制取得（未定義ならエラー）
        self.cutoff = self.data.get("CutOffVDW", None)
        if self.cutoff is None:
            raise ValueError("CutOffVDW is not defined in force_field.json")

        # -----------------------------
        # parse
        # -----------------------------
        self.pseudo_atoms = self._parse_pseudo_atoms()
        self.lj_params = self._parse_self_interactions()

        # -----------------------------
        # build
        # -----------------------------
        self._build_type_map()
        self._build_type_categories()
        self._build_interaction_matrices()

        self._validate()

    # =================================================
    # Parsing
    # =================================================

    def _parse_pseudo_atoms(self):
        atoms = {}

        for entry in self.data["PseudoAtoms"]:
            name = entry["name"]

            atoms[name] = {
                "element": entry.get("element"),
                "mass": entry.get("mass"),
                "charge": entry.get("charge", 0.0),
                "framework": entry.get("framework", False),
            }

        return atoms

    def _parse_self_interactions(self):

        lj_dict = {}

        for entry in self.data["SelfInteractions"]:

            if entry["type"].lower() != "lennard-jones":
                continue

            name = entry["name"]

            epsilon_K, sigma_A = entry["parameters"]

            # ★ 単位はそのまま固定（K, Å）
            lj_dict[name] = {
                "epsilon": float(epsilon_K),
                "sigma": float(sigma_A),
            }

        return lj_dict

    # =================================================
    # Build
    # =================================================

    def _build_type_map(self):

        # ★ 順序固定（重要）
        self.type_list = sorted(self.pseudo_atoms.keys())

        self.type_map = {name: i for i, name in enumerate(self.type_list)}
        self.n_types = len(self.type_list)

    def _build_type_categories(self):

        self.framework_types = [t for t in self.type_list if self.is_framework(t)]

        self.adsorbate_types = [t for t in self.type_list if not self.is_framework(t)]

        self.framework_type_ids = [self.type_map[t] for t in self.framework_types]

        self.adsorbate_type_ids = [self.type_map[t] for t in self.adsorbate_types]

        # ★ 吸着種が存在しない場合は即エラー
        if len(self.adsorbate_type_ids) == 0:
            raise ValueError("No adsorbate types found")

    def _build_interaction_matrices(self):

        n = self.n_types

        self.eps_matrix = np.zeros((n, n))
        self.sig_matrix = np.zeros((n, n))

        for a in self.type_list:
            for b in self.type_list:

                i = self.type_map[a]
                j = self.type_map[b]

                eps, sig = self._mix_lj_pair(a, b)

                self.eps_matrix[i, j] = eps
                self.sig_matrix[i, j] = sig

    # =================================================
    # Validation（重要）
    # =================================================

    def _validate(self):

        # --- LJ未定義チェック ---
        missing = []
        for atom in self.pseudo_atoms:
            if atom not in self.lj_params:
                missing.append(atom)

        if len(missing) > 0:
            raise ValueError(f"LJ params missing for: {missing}")

        # --- 値の異常チェック ---
        if np.any(self.sig_matrix <= 0):
            raise ValueError("Sigma contains non-positive values")

        if np.any(self.eps_matrix < 0):
            raise ValueError("Epsilon contains negative values")

    # =================================================
    # Public API
    # =================================================

    def get_lj(self, atom_type):
        params = self.lj_params[atom_type]
        return params["epsilon"], params["sigma"]

    def is_framework(self, atom_type):
        return self.pseudo_atoms[atom_type]["framework"]

    def get_cutoff(self):
        return self.cutoff  # Å

    # =================================================
    # Mixing
    # =================================================

    def _mix_lj_pair(self, atom_i, atom_j):

        eps_i, sig_i = self.get_lj(atom_i)
        eps_j, sig_j = self.get_lj(atom_j)

        if self.mixing_rule.lower() == "lorentz-berthelot":

            sigma_ij = 0.5 * (sig_i + sig_j)
            epsilon_ij = np.sqrt(eps_i * eps_j)

            # debug
            # print(f"[DEBUG][RaspaForceField] sigma_ij: {sigma_ij}")
            # print(f"[DEBUG][RaspaForceField] epsilon_ij: {epsilon_ij}")

        else:
            raise NotImplementedError

        return epsilon_ij, sigma_ij


# class RaspaForceField:
#     """
#     Responsibilities:
#     Parser and container for RASPA3 force_field.json

#     設計メモ：
#     これ自体で「いい感じにデータを揃えてくれる」類の装置ではない。
#     パーサとして、他のクラスから呼び出される形で役立つ。


#     """

#     def __init__(self, json_path):
#         with open(json_path, "r") as f:
#             self.data = json.load(f)

#         self.mixing_rule = self.data.get("MixingRule", "Lorentz-Berthelot")
#         self.truncation_method = self.data.get("TruncationMethod", None)
#         self.tail_corrections = self.data.get("TailCorrections", False)
#         self.cutoff = self.data.get("CutOffVDW", None)

#         self.pseudo_atoms = self._parse_pseudo_atoms()
#         self.lj_params = self._parse_self_interactions()

#         # for energy grid calculation
#         self._build_type_map()  # 現行では外部にあるtype_mapping関数に相当する機能を本クラスに移植して、現type_mappingは封印
#         self._build_type_categories()
#         self._build_interaction_matrices()

#         self._validate()

#     # -------------------------------------------------
#     # Parsing
#     # -------------------------------------------------

#     def _parse_pseudo_atoms(self):
#         atoms = {}

#         for entry in self.data["PseudoAtoms"]:
#             name = entry["name"]
#             atoms[name] = {
#                 "element": entry.get("element"),
#                 "mass": entry.get("mass"),
#                 "charge": entry.get("charge", 0.0),
#                 "framework": entry.get("framework", False),
#             }

#         return atoms

#     def _parse_self_interactions(self):
#         lj_dict = {}

#         for entry in self.data["SelfInteractions"]:
#             if entry["type"].lower() != "lennard-jones":
#                 continue

#             name = entry["name"]

#             epsilon_K, sigma_A = entry["parameters"]

#             # 単位変換禁止
#             lj_dict[name] = {"epsilon": epsilon_K, "sigma": sigma_A}  # ← Kelvinのまま  # ← Åのまま

#         return lj_dict

#     def _build_type_map(self):
#         """
#         to build type map from Raspa's json data.

#         """
#         self.type_list = list(self.pseudo_atoms.keys())

#         self.type_map = {name: i for i, name in enumerate(self.type_list)}

#         self.n_types = len(self.type_list)

#     def _build_type_categories(self):
#         """
#         to separate type categories:
#             - framework(absorbent ex.MOF)
#             - adsorbate(ex. CO2 gas)

#         """
#         self.framework_types = [t for t in self.type_list if self.is_framework(t)]

#         self.adsorbate_types = [t for t in self.type_list if not self.is_framework(t)]

#         self.framework_type_ids = [self.type_map[t] for t in self.framework_types]

#         self.adsorbate_type_ids = [self.type_map[t] for t in self.adsorbate_types]

#     def _build_interaction_matrices(self):
#         """
#         to build
#         """
#         n = self.n_types

#         self.eps_matrix = np.zeros((n, n))
#         self.sig_matrix = np.zeros((n, n))

#         for a in self.type_list:
#             for b in self.type_list:

#                 i = self.type_map[a]
#                 j = self.type_map[b]

#                 eps, sig = self._mix_lj_pair(a, b)

#                 self.eps_matrix[i, j] = eps
#                 self.sig_matrix[i, j] = sig

#     def _validate(self):
#         for atom in self.pseudo_atoms:
#             if atom not in self.lj_params:
#                 print(f"Warning: No LJ parameters found for atom type {atom}")

#     # -------------------------------------------------
#     # Public API
#     # -------------------------------------------------

#     def get_atom_types(self):
#         return list(self.pseudo_atoms.keys())

#     def is_framework(self, atom_type):
#         return self.pseudo_atoms[atom_type]["framework"]

#     def get_charge(self, atom_type):
#         return self.pseudo_atoms[atom_type]["charge"]

#     def get_mass(self, atom_type):
#         return self.pseudo_atoms[atom_type]["mass"]

#     def get_lj(self, atom_type):
#         """
#         Returns epsilon (K) and sigma (Å)
#         """
#         params = self.lj_params[atom_type]
#         return params["epsilon"], params["sigma"]

#     def _minimum_image(self, d, box_length):
#         return d - box_length * np.round(d / box_length)

#     # -------------------------------------------------
#     # Mixing rules
#     # -------------------------------------------------

#     def _mix_lj_pair(self, atom_i, atom_j):
#         """
#         to mix epsilon and sigma (i, j) pairs.
#         """
#         eps_i, sig_i = self.get_lj(atom_i)
#         eps_j, sig_j = self.get_lj(atom_j)

#         if self.mixing_rule.lower() == "lorentz-berthelot":

#             sigma_ij = 0.5 * (sig_i + sig_j)
#             epsilon_ij = np.sqrt(eps_i * eps_j)

#         else:
#             raise NotImplementedError

#         return epsilon_ij, sigma_ij

#     # -------------------------------------------------
#     # Utility
#     # -------------------------------------------------
#     def framework_atom_types(self):
#         return [a for a in self.pseudo_atoms if self.is_framework(a)]

#     def adsorbate_atom_types(self):
#         return [a for a in self.pseudo_atoms if not self.is_framework(a)]

#     def summary(self):
#         print("=== RASPA Force Field Summary ===")
#         print("Mixing rule:", self.mixing_rule)
#         print("Cutoff (Å):", self.cutoff)
#         print("Tail corrections:", self.tail_corrections)
#         print("\nAtom types:")

#         for atom in self.pseudo_atoms:
#             eps, sig = self.get_lj(atom)
#             print(f"{atom:10s}  eps={eps:10.3f} J/mol   sigma={sig:6.3f} Å")


### --- 使用例 --- ###
if __name__ == "__main__":
    pass
    # より洗練された設計：行列を作る（行列演算の方が後々便利）
    # Ex. energy = 4 * eps_matrix * (...)

    # import pprint

    # ff = RaspaForceField("force_field.json")

    # ff.summary()

    # eps, sig = ff.get_lj("O1")

    # eps_mix, sig_mix = ff.mix_lj("O1", "O_co2")

    # framework_atoms = ff.framework_atom_types()
    # adsorbate_atoms = ff.adsorbate_atom_types()

    # 吸着質-骨格原子の全epx, sigの組み合わせを取得

    # # 行列の次元を決定
    # n_ads = len(adsorbate_atoms)
    # n_frm = len(framework_atoms)
    # d_mat = n_ads + n_frm

    # # key_mapも自動生成でなければ意味がない
    # type_list = adsorbate_atoms + framework_atoms
    # type_map = dict()

    # for i, type_id in enumerate(type_list):
    #     type_map[type_id] = i

    # # type_mapの中身確認
    # pprint.pprint(type_map)
    # print()

    # eps_matrix = np.zeros((d_mat, d_mat))
    # sig_matrix = np.zeros((d_mat, d_mat))

    # for atom_i in type_list:
    #     for atom_j in type_list:
    #         i = type_map[atom_i]
    #         j = type_map[atom_j]
    #         eps_matrix[i, j], sig_matrix[i, j] = ff.mix_lj(atom_i, atom_j)  # データの実体はffに入っている

    # print(eps_matrix.shape)

    # # pprint.pprint(eps_matrix)

    # # 実務では以下のように取り出す
    # # クロス積のみの場合：
    # for atom_i in adsorbate_atoms:
    #     for atom_j in framework_atoms:
    #         print(f"eps[{atom_i}, {atom_j}]:{eps_matrix[type_map[atom_i], type_map[atom_j]]}")

    # print()

    # # 全原子の組み合わせの場合:
    # for atom_i in type_list:
    #     for atom_j in type_list:
    #         print(f"eps[{atom_i}, {atom_j}]:{eps_matrix[type_map[atom_i], type_map[atom_j]]}")
