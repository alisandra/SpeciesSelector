# Species Selection Example
This project is not designed to be portable or reusable, 
merely to organize and document the way final
training species selection was performed for the Helixer project.
Additionally it serves as notes should Helixer project members desire
to re-run species selection in the future.

## Pre-filter by metadata (optional)

```bash
spselec --config example/config.ini metafilter
```

## Setup and start training of first seeds
Make sure all the paths are right in the config.
- The species full/subset are the .h5 files (think Helixer's export).
- The tree file is from ncbi via ete3. 
- NNI config has the main training parameters etc, and most machine-specific
  config hard coded (mode control or load model path will be appended)
- Working dir is the structured working and output directory (empty or created)

```bash
spselec --config example/config.ini setup
# or if you're using the pre-filtering (from above), change to:
# spselec --config example/config.ini setup --passed-meta-filter \
#  /home/ali/repos/github/alisandra/SpeciesSelector/example/passed_meta_filter.txt
```
This splits the species into halves, selects random seeds for each,
setups up a whole round of nni files, and starts the first training
of seed models. Everything from current trainers, to models and 
results, to project status are recorded in a sqlite3 db in
the working directory.

## Run next step
Run this whenever the nni jobs have fully finished
```bash
spselec --config example/config.ini next
```
This will check the status, record any results, make
any adjustments such as adding or removing training species
at the end of a round based upon the X-split performance,
and finally (prep) and submit the next step to nni. 

Iterate as long as desired (e.g. 3 rounds), there is no
official end condition nor command nor detection of convergence
(presumably takes too long). There are some forced end
states, e.g. _all_ species have been selected for training
or _all_ for validation. 

## Notes

### Resuming
This is not (yet) particularly automated for resuming.
Nevertheless it is usually possible.

- if you have to restart the nni-job completely, update the db nni experiment id as appropriate
- if you just had to re-run the trials. E.g. `nohup bash run.sh`; then just use
  `nnictl view <experiment ID> -p <expected port for split>` so that 
  `nnictl stop <experiment ID>` can run. 
- file and symlink creation should mostly be robust / tolerate a rerun.
- depending on when and what hapend/how much should be re-ran it may be 
  necessary to manually set db status (or implement a rollback)

### Config
If a config file should be silently recognized
(e.g. if `--tuner-gpu-indices` is in use, and remembering `--config`
is annoying for every `spselec next` iteration) 
just copy the config to $HOME/.config/spselec/config.ini