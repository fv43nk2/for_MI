### --- import block --- ###
import numpy as np


### --- class block --- ###
class EnergyDistributor:
    """
    Responsibilities:
    - generate energy distribution for GCMC calc.

    Args:
        ----------
        N_sites : int
            Total number of adsorption sites

        seed : int or None
            Random seed for reproducibility

    Returns
        -------
        eps : np.ndarray
            Array of shape (N_sites,)

    設計メモ：
        - strong/weakにそれぞれ紐づく吸着エネルギーの期待値・分散は、いわば吸着材(MOF)の固有物性。
        ⇒ これをparams(dict)として与えるのが筋。

    - 本クラスはparmsの中身を見ない。別クラスから渡されるままに展開して使うだけ
        ⇒ 差し当たり、parmsの形だけ明確に決めておく。メタ的なやつから、物性まで。

        - 次ステップ：骨格由来のエネルギー分布(ε_framework)
            + ガス由来のエネルギー分布(ε_gas)という考え方を導入
            ⇒ 物理モデルの設計は別クラス（EnergyModelなど）に任せる

    """

    def __init__(self, params: dict):
        """
        Parameters
        ----------
        params : dict
            Energy distribution settings.

            Expected format:
            {
                "mode": "dual",

                "sites_distribution": {
                    "strong": 0.3,
                    "weak": 0.7,
                },

                "sites": {
                    "strong": {
                        "energy_mu": -30000.0,
                        "energy_sigma": 2000.0,
                    },
                    "weak": {
                        "energy_mu": -15000.0,
                        "energy_sigma": 2000.0,
                    }
                }
            }
        """

        self.params = params
        self.mode = params.get("mode", "single")

        self._validate_params()

    # ==========================================================
    # Public API
    # ==========================================================

    def generate_energy_distribution(self, N_sites: int = 2000, seed: int = None):
        """
        Generate adsorption energy distribution ε.

        Parameters
        ----------
        N_sites : int
            Total number of adsorption sites

        seed : int or None
            Random seed for reproducibility

        Returns
        -------
        eps : np.ndarray
            Array of shape (N_sites,)
        """

        if seed is not None:
            np.random.seed(seed)

        if self.mode == "single":
            return self._generate_single(N_sites)

        elif self.mode == "dual" or self.mode == "multi":
            return self._generate_multi(N_sites)

        elif self.mode == "broad":
            return self._generate_broad(N_sites)

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    # ==========================================================
    # Internal Generators
    # ==========================================================

    def _generate_single(self, N_sites):
        """
        Single energy value (degenerate case)
        """

        site_name = list(self.params["sites"].keys())[0]
        mu = self.params["sites"][site_name]["energy_mu"]

        return np.full(N_sites, mu)

    def _generate_multi(self, N_sites):
        """
        Multi-site Gaussian mixture
        """

        site_dist = self.params["sites_distribution"]
        sites = self.params["sites"]

        eps_list = []

        for site_name, fraction in site_dist.items():
            n = int(N_sites * fraction)

            mu = sites[site_name]["energy_mu"]
            sigma = sites[site_name]["energy_sigma"]

            eps = np.random.normal(mu, sigma, n)
            eps_list.append(eps)

        eps_all = np.concatenate(eps_list)

        # サイズ補正（丸め誤差対策）
        if len(eps_all) < N_sites:
            deficit = N_sites - len(eps_all)
            mu = np.mean(eps_all)
            eps_extra = np.random.normal(mu, 1000.0, deficit)
            eps_all = np.concatenate([eps_all, eps_extra])

        np.random.shuffle(eps_all)

        return eps_all

    def _generate_broad(self, N_sites):
        """
        Single broad Gaussian distribution
        """

        mu = self.params.get("broad_mu", -22000.0)
        sigma = self.params.get("broad_sigma", 7000.0)

        return np.random.normal(mu, sigma, N_sites)

    # ==========================================================
    # Validation
    # ==========================================================

    def _validate_params(self):
        """
        Minimal validation for robustness
        """

        if self.mode in ["dual", "multi"]:
            if "sites_distribution" not in self.params:
                raise ValueError("Missing sites_distribution")

            if "sites" not in self.params:
                raise ValueError("Missing sites block")

            total = sum(self.params["sites_distribution"].values())
            if not np.isclose(total, 1.0):
                raise ValueError("sites_distribution must sum to 1.0")

        if self.mode == "single":
            if "sites" not in self.params:
                raise ValueError("Single mode requires sites definition")
