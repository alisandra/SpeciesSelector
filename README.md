# SpeciesSelector

>This code is released primarily for the purpose of documentation. 

Implements 2-fold cross validation for which species to train with for the Helixer 
deep learning gene calling tool. This repository:
  - sets up data
  - starts training and evaluation of Helixer modesl via `nnictl`
  - summarizes results. 
  
Note that while the code allows extended iteration and refinement of 
a promising set, in practice compute was better spent in one round with many 
random starts and then a remix. 

General process is as follows

```
# setup data & start many training runs with randomly selected 
spselec setup <parameters>

# for as many rounds as makes sense (stop at status 'seeds evaluating', probably once for recommended half round only)
# repeat the following (run after previous round finishes, manual check requried)
spselec next <parameters>

# when ready
# combine training species from best two models in each fold, and all other species as validation
spselec remix-train

# evaluate on all species that were in none of the remix training sets above
spselec remix-eval
```
