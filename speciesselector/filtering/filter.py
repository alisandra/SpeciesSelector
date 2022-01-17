import pandas
import pandas as pd
from collections import defaultdict
from speciesselector.helpers import match_tree_names_plants, match_tree_names_exact
import ete3


class Filter:
    def __init__(self, dat, threshold, exempt=None):
        self.dat = dat.copy()
        self.threshold = threshold
        self.exempt = exempt

    def filter(self, remaining):
        out = []
        mask = self._species_that_pass()
        passed = set(self.dat.loc[mask, 'species'])
        for r in remaining:
            if r in passed:
                out.append(r)
            elif r in self.exempt:
                out.append(r)
        return out

    def _species_that_pass(self):
        raise NotImplementedError


class UTRFilter(Filter):
    def _species_that_pass(self):
        dat = self.dat
        dat.loc[dat['longest_UTR__count'].isna(), 'longest_UTR__count'] = 0
        utr_frac = dat['longest_UTR__count'] / dat['longest_mRNA__count'] / 2
        return utr_frac > self.threshold


class AltSpliceFilter(Filter):
    def _species_that_pass(self):
        dat = self.dat
        dat.loc[dat['longest_UTR__count'].isna(), 'longest_UTR__count'] = 0
        # -1 so that it is staying ball-park comparable in filtering numbers
        alternative_frac = dat['all_mRNA__count'] / dat['longest_mRNA__count'] - 1
        return alternative_frac > self.threshold


class BuscoFilter(Filter):
    def _species_that_pass(self):
        busco_loss = self.dat['busco_C_prot'] - self.dat['busco_C_geno']
        print(busco_loss)
        # threshold is s.t. like 0.05, for 5% LOSS of genes between genome and proteome
        return busco_loss > -self.threshold


def exemptions_as_sp_names(exemption_dict, sp_names, tree_path, exact_match=False):
    tree = ete3.Tree(tree_path)
    tree_names = [x.name for x in tree.get_leaves()]
    if exact_match:
        match_fn = match_tree_names_exact
    else:
        match_fn = match_tree_names_plants
    t2skey = match_fn(tree_names, sp_names)

    # typical exemption_dict: {"altsplice": ["Porphyra umbilicalis - 2786", "Chlorophyta - 3041"]}
    # to figure out exact names, use ete3.Tree.show()!
    out = {"busco": set(), "altsplice": set(), "UTR": set()}
    for rule, exemptions in exemption_dict.items():
        for exempt in exemptions:
            print(f'exempting : {exempt} \nfrom rule: {rule}')
            exempt_found = [x for x in tree.get_descendants() if x.name == exempt]
            assert len(exempt_found) == 1, "len(exempt_found) == {} != 1... {}".format(len(exempt_found), exempt_found)
            exempt_found = exempt_found[0]
            x_species = [x.name for x in exempt_found.get_leaves()]
            x_species = [t2skey[x] for x in x_species]  # convert back to names matching 'species'
            x_species = set(x_species)
            print(x_species, '<- exempted')
            out[rule].update(x_species)
    return out
