# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 11:05:56 2023

@author: ivanp
"""

import json

## randomizer sequence

SUBSTITUENTS = ["acetyl","bromo","hydroxy","benzoyl",
                "cyano","ethoxy","methoxy","phenyl",
                "ethoxycarbonyl","methoxycarbonyl","dihydroxyamino"]
molecule_data = {"molecule_1":
                     {"base":"coumarin",
                      "positions":[3,4,5,6,7],
                      "min_sub":1,
                      "max_sub":3,
                      "subs":SUBSTITUENTS,
                      "n_molecules":1000
                     },
                    }

## molecule descriptors
# preprocessing features before genetic algorithm
pre_params = {"correlation_limit":0.95,
                    "repeated_value_limit":0.80,
                    "activity_sampling_groups":2,
                    "rng":5,
                    "alpha":0.05,
                    "top_percentile":50,
                    "k":None,
                    "extra_genes":0,
                    }

# parameters of genetic algorithm
genalg_params = {"n_cores" : 1,
   "npoint" : 2,
   "mutating_individuals" : 1,
   "mutation_probability":0.05,
   "eugenic_generation_limit" :10,
   "eugenic_mixture" : 0.5,
   "gen_limit" : 50,
   "pop_size" : None,
   "start_gene":None
   }

# # parameters of unifying models
post_params = {"min_fitness":0.5,
            "alpha":0.05,
            "alternative":"greater",
            "y":2,
            "method":"holm"
            }


with open("config/randomizer_parameters.json", "w") as write_file:
    json.dump(molecule_data, write_file)

with open("config/model_parameters.json","w") as write_file:
    json.dump({"pre_params":pre_params,
               "genalg_params":genalg_params,
               "post_params":post_params},
              write_file)
