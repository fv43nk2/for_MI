### --- import block --- ###
import numpy as np
import re
from pymatgen.core import Structure
from typing import Any


### --- method block --- ###
class SimulationStructure:
    """
    Responsibilities:

    Args:

    Returns:

    設計メモ:
    -

    """

    # 単位換算用
    angstrom_to_meter = 1e-10

    def __init__(self, pymatgen_structure, force_field):

        self.structure = pymatgen_structure
        self.ff = force_field

        self.positions = pymatgen_structure.cart_coords * self.angstrom_to_meter

        self.box = np.array(pymatgen_structure.lattice.abc) * self.angstrom_to_meter

        self.framework_types = self._map_atom_types()

    ### --- public API block start --- ###
    def __call__(self):

        # 1. 座標 (N, 3)
        positions = self.structure.cart_coords  # デフォルトでÅ単位のデカルト座標

        # 格子情報 (3x3行列)
        lattice = self.structure.lattice.matrix

        # --- 3. グリッド作成  ---
        box = list(
            abc * self.angstrom_to_meter for abc in self.structure.lattice.abc
        )  # 26.3... つまりÅ表記なので単位換算必須

        angles = list(ang for ang in self.structure.lattice.angles)

        # box =+ angles

        structure_data = {
            "framework_positions": positions,  # np.ndarray
            "framework_lattice": lattice,  # np.ndarray おそらく不要
            "box": box,  # list
            "angles": angles,  # float 今後の拡張で分子の接近方向を考慮する必要が出てきた場合に対処
            "framework_types": self.framework_types,
        }

        return structure_data

    ### --- public API block end --- ###

    ### --- private method block --- ###
    def _map_atom_types(self):

        type_ids = []

        for site in self.structure:
            atom_type = site.label  # Raspaのcifファイルに記述される element ⇒ ffのtypeは1対他(typeの方が多)
            type_id = self.ff.type_map[atom_type]
            type_ids.append(type_id)

        return np.array(type_ids)


class CifToStructure:
    """
    Responsibilities:

    Args:

    Returns:

    設計メモ:
    - for ループで2つ以上のcifデータを読み込むことはあるのではないか？
        ⇒ であれば、インスタンス化して再利用する可能性があるかも

    - structure_dataという形でcifからの構造データをまとめるのであれば、別に関数でいい気がする
    """

    def cif_to_structure(self, cif_path):

        struct = Structure.from_file(cif_path)

        return struct

    # def cif_to_structure_data(self, cif_path) -> dict[str, np.ndarray | tuple]:
    #     """
    #     Args:
    #     - cif_path

    #     Returns:
    #     -

    #     """
    #     # --- file load --- ###
    #     struct = Structure.from_file(cif_path)

    #     # 1. 座標 (N, 3)
    #     positions = struct.cart_coords  # デフォルトでÅ単位のデカルト座標

    #     # 格子情報 (3x3行列)
    #     lattice = struct.lattice.matrix

    #     # --- 3. グリッド作成  ---
    #     box = list(abc * self.angstrom_to_meter for abc in struct.lattice.abc)  # 26.3... つまりÅ表記なので単位換算必須

    #     angles = list(ang for ang in struct.lattice.angles)

    #     # box =+ angles

    #     structure_data = {
    #         "framework_positions": positions,  # np.ndarray
    #         "framework_lattice": lattice,  # np.ndarray おそらく不要
    #         "box": box,  # list
    #         "angles": angles,  # float 今後の拡張で分子の接近方向を考慮する必要が出てきた場合に対処
    #     }

    #     return structure_data
