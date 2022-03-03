import re


def split_list(split):
    """splits to run as list"""
    if split is None:
        return [0, 1]
    else:
        assert isinstance(split, int)
        return [split]


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


class F1Decode:
    IG = 'ig'
    UTR = 'utr'
    CDS = 'exon'
    INTRON = 'intron'
    GENIC = 'genic'
    CATEGORIES_OF_INTEREST = [IG, UTR, CDS, INTRON, GENIC]


# parsing trial.log/eval.log files
def parse_eval_log(log_file_path):
    # each line in the table will have ['name', 'norm.H', 'Precision', 'Recall', 'F1-Score']
    f1_ret = {}
    for table in gen_tables(log_file_path):
        header, tab = parse_table(table)
        if header == "F1_summary":
            for sline in tab:
                if sline[0] in F1Decode.CATEGORIES_OF_INTEREST:
                    f1_ret[sline[0]] = sline[4]
    return f1_ret


# next two functions modified from ascii_scores_to_csv from Helixer/scripts
def parse_table(splittable):
    out = []
    # splittable = table.split('\n')
    header = splittable[0]
    header = re.sub('\+|-', '', header)
    for line in splittable[1:]:
        if not line.startswith('+'):
            line = line.replace(' ', '')
            line = re.sub('\|', ',', line)
            sline = line.split(',')
            # drop empty start/end values
            if not sline[0]:
                sline = sline[1:]
            if not sline[-1]:
                sline = sline[:-1]
            out.append(sline)

    return header, out


def gen_tables(filein):
    out = []
    with open(filein) as f:
        for line in f:
            line = line.rstrip()
            line = re.sub('\[.*\] PRINT ', '', line)  # crop front logging timestamp +, if present
            if line.startswith('|') or line.startswith('+'):
                out.append(line)
            else:
                # ignore double non-table lines etc...
                if out:
                    yield out
                    out = []
    if out:
        yield out


def divvy_up_gpu_indices(raw_indices):
    """take command-line indices (printer page style, e.g. 1,45,7-9) and split into two explicit halves 1,7 & 8,9,45"""
    # returned as a string, to directly fill the gpuIndices NNI parameter
    if raw_indices is None:
        return None, None
    raw_numbers = []
    items = raw_indices.split(',')
    for item in items:
        if item.find('-') > -1:
            start, end = [int(x) for x in item.split('-')]
            raw_numbers += list(range(start, end + 1))
        else:
            raw_numbers += [int(x) for x in raw_numbers]
    raw_numbers = list(set(raw_numbers))
    assert len(raw_numbers) >= 2
    halfway = len(raw_numbers) // 2
    return int_list_to_str(raw_numbers[:halfway]), int_list_to_str(raw_numbers[halfway:])


def int_list_to_str(ints):
    return ','.join([str(i) for i in ints])