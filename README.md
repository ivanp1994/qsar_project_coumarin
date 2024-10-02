The project for the "Modeling biologically important molecules and their complexes" [course](https://mobi.unios.hr/en/predmeti/modeliranje-bioloski-vaznih-molekula-i-njihovih-kompleksa-2204/) of the Bioinformatics module.

The task of this project was to retrace the steps of the paper [Coumarin Derivatives Act as Novel Inhibitors of Human Dipeptidyl Peptidase III: Combined In Vitro and In Silico Study](https://www.mdpi.com/1424-8247/14/6/540) related to QSAR.
Briefly, the paper studies the ability of coumarin derivates to inhibit Dipeptidyl Peptidase III. Based on this ability, they use principles of QSAR to search for another batch of coumarin derivates that have high hDP3 inhibitory potential.
The paper uses commercial QSAR software (Avogardo and QSARINS) and the goal of this project was:

-1 Develop an open-source analog to said software 
-2 Create a reproducible pipeline capable of being used on different molecules 

The data table used is found in `config/experimental_data.csv` and consists of two columns - first being the compound name and the other being an experimental value. 
This repository performs the following:

-1 Converts compounds in [SMILES](https://en.wikipedia.org/wiki/Simplified_Molecular_Input_Line_Entry_System) representation using Cactus WEB service. 
-2 Creates randomized new molecules based on a defined compound. The parameters for creating new molecules are found in `config/randomizer_paramters.json`. The parameters can also be created by modifying `defining_parameters.py` script.
-3 Converts molecules in SMILES form into molecule descriptors (both 2D and 3D) using `mordred` package.
-4 Uses a four different machine learning models to predict activity (in our case, inhibitory ability).

## Predicting activity
For splitting data (molecules) into training and validation sets, we implement Activity Sampling. Briefly, there is a lot of molecules that are chemically inert, and this can have adverse consequences on model training.
[Activity Sampling](https://pubmed.ncbi.nlm.nih.gov/13677490/) is splitting training and test sets based on their activity so that the inert molecules do not dominate either set. We split molecules into 3 sets - Inert (activity is 0), Low, and High. Set of test molecules contains at least one molecule from each set.
Our implementation of Activity Sampling is found in `workflow/scripts/activity_sampling.py`.

The problem with using molecule descriptors in QSAR modeling is that a bunch of descriptors are highly correlated, or repeating within the set. Furthermore, not all descriptors are (or should be) used.
We drop highly correlated and low variance descriptors, then implement feature selection via `SelectKBest` of sklearn model to reduce the number of descriptors to a manageable number.
We employ a genetic algorithm approach to selecting descriptors - briefly, we train several models using several different descriptors and then "breed" them according to their fitness.
This process is iterated until a set end (several generations passes, fitness limit goes above a defined limit, a population "converges" in genetic variance, etc). The GeneticAlgorithm implementation along with parent selection is found in `workflow/scripts/genetic_algorithm.py`.


We employ 4 different ML models - Partial Squares Linear Regression, LinearRegression (with Ridge and Lasso regularization) and join the models to find which one of "fictive" molecules
consistently show a high level of hDP3 inhibition. 

Our models are wrapped in snakemake pipeline. Sadly, `mordred` is limited to and below Python 3.6, so two different environs are used. Conda environs are found in `envs` folder.

