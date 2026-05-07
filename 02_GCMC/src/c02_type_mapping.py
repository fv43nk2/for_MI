### --- import block --- ###


### --- method block --- ###
def type_mapping(adsorbate_atoms, framework_atoms):
    """
    Returuns:

    """

    # key_mapも自動生成でなければ意味がない
    type_list = adsorbate_atoms + framework_atoms
    all_type_map = dict()

    ### --- 全原子を一律管理 --- ###
    for i, type_id in enumerate(type_list):
        all_type_map[type_id] = i

    ### ---クロス積専用に個別管理 --- ###
    # --- adsorbate --- #
    ads_map = {k: v for k, v in all_type_map.items() if k in adsorbate_atoms}

    # --- framework --- #
    frm_map = {k: v for k, v in all_type_map.items() if k in framework_atoms}

    type_map_dict = {
        "all_type_map": all_type_map,
        "adsorbate_map": ads_map,
        "framework_map": frm_map,
    }

    return type_map_dict
