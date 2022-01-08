def match_tree_names_exact(tree_names, sp_dirs):
    """makes sure they __all__ match exactly"""
    t2skey = {}
    sp_dirs = sp_dirs.copy()
    for tn in tree_names:
        if tn in sp_dirs:
            sp_dirs.remove(tn)
            t2skey[tn] = tn
        else:
            raise ValueError(f"{tn} could not be exact matched to any remaining option: {sp_dirs}")
    assert not len(sp_dirs), "unmatched species remaining"
    return t2skey


def match_tree_names_plants(tree_names, sp_dirs):
    """matches between Gspecies and 'genus species -- taxid', with some noise tolerance (as necessary for plants)"""
    sp_dirs = sp_dirs.copy()
    sp_to_tree = {}
    for tn in tree_names:
        gspecies = tn[0] + tn.split()[1]
        if gspecies in sp_dirs:
            sp_to_tree[gspecies] = tn
            sp_dirs.remove(gspecies)
        else:
            if gspecies.endswith('.'):
                gspecies = gspecies[:-1]
            match_at_start = [x for x in sp_dirs if x.startswith(gspecies)]
            if len(match_at_start) == 1:
                sp_to_tree[match_at_start[0]] = tn
                sp_dirs.remove(match_at_start[0])
            else:
                raise ValueError(f"{tn} could not be matched to any remaining option: {sp_dirs}")
    assert not len(sp_dirs), "unmatched species remaining"
    t2skey = {t: s for s, t in sp_to_tree.items()}
    return t2skey
