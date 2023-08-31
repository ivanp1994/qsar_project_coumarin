# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 14:33:21 2023
https://stats.stackexchange.com/questions/179733/theory-behind-partial-least-squares-regression

@author: ivanp
"""
import argparse
import json
import numpy as np
import pandas as pd
import os
os.environ["OUTDATED_IGNORE"] = "1"
from functools import partial


from scipy import stats
import pingouin as pg
from sklearn.cross_decomposition import PLSRegression
import sklearn.linear_model as lm
from sklearn.feature_selection import SelectKBest,f_regression
from genetic_algorithm import GeneticAlgorithm, parent_selection_rulette
from activity_sampling import ActivitySampler, CrossValidator

import logging

MODEL_DICTIONARY = {"pls":PLSRegression(),
                    "ridge":lm.Ridge(),
                    "lasso":lm.Lasso(),
                    "linear":lm.LinearRegression()
                    }
PARENT_SELECTION = {"rulette":parent_selection_rulette,
                    }

# loading data from command line
parser = argparse.ArgumentParser()
parser.add_argument('-f')#, default="molecules/described/old_molecules.csv")
parser.add_argument('-y')#, default="config/experimental_data.csv")
parser.add_argument('-new')#, default="molecules/described/new_molecules.csv")
parser.add_argument('-threads',type=int)#,default=8)
parser.add_argument('-p')#, default="config/model_parameters.json")
parser.add_argument('-mod_pred')#, default="molecules/models/lasso_predicted.csv")
parser.add_argument('-mod_pars')#, default="molecules/models/lasso_parameters.csv")

args = vars(parser.parse_args())

target = pd.read_csv(args["y"],header=None,index_col=0)
features = pd.read_csv(args["f"],index_col=0)
random_molecules = pd.read_csv(args["new"],index_col=0)

with open(args["p"], "r") as _f:
    _model_parameters= json.load(_f)

pre_params = _model_parameters["pre_params"]
genalg_params = _model_parameters["genalg_params"]
post_params = _model_parameters["post_params"]

threads = args.get("threads",None)
if threads is not None:
    genalg_params["n_cores"] = threads


output_models = args["mod_pars"]
output_predicted = args["mod_pred"]
model = output_predicted.split("/")[-1].split("_")[0]
logging.basicConfig(level=logging.INFO,format='%(message)s',filename=f"{model}.log")
logging.getLogger().addHandler(logging.StreamHandler())



#%% FUNCTIONS


def drop_intercorrelated(df,corr=0.95,inplace=False):
    """
    drops values that are too intercorrelated from
    the dataframe
    """
    corr_matrix = df.corr().abs()
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    # Find features with correlation greater than 0.95
    to_drop = [column for column in upper.columns if any(upper[column] > corr)]
    if inplace:
        return df.drop(to_drop, axis=1, inplace=True)
    return df.drop(to_drop,axis=1)

def drop_repeating_values(df,limit=0.85,inplace=False):
    """
    drops columns that have values that are repeated too often.
    for example, for limit 0.85 it drops columns when a value in a
    column occurs at least 85%

    the alogrithm is interesting
    first we get the number of unique values
    then we get real limit which is limit* length of our dataframe
    for example, when we have a dataframe of 100, and limit of 0.85,
    then we can theoretically have 85 repeats and 15 unique values
    for a total of (15 + 1) unique values.
    Meaning that we only need to look at columns that have
    less than (0.85*100)+1 unique files.
    """
    real_limit = int(limit*len(df))+1
    unique_values = df.T.apply(lambda x: x.nunique(),axis=1).sort_values()
    suspect_columns = unique_values[unique_values<real_limit].index.tolist()
    suspect_df = df[suspect_columns]
    suspect_series = suspect_df.apply(lambda col: col.value_counts().max()/len(col),axis=0)
    to_drop = suspect_series[suspect_series>limit].index.tolist()

    if inplace:
        return df.drop(to_drop, axis=1, inplace=True)
    return df.drop(to_drop,axis=1)

def select_top_features(features,target,k=None,top_percentile=None,alpha=0.05):
    "selects bet features "
    selector = SelectKBest(f_regression,k="all")
    selector.fit(features,target)

    scores = selector.scores_
    pvals = selector.pvalues_

    column_data = pd.DataFrame([scores,pvals],index=["scores","pvals"]).T
    filtered = column_data[column_data.pvals<alpha].sort_values("scores",ascending=False)
    if k is not None and top_percentile is None:
        limit = k
    elif k is None and top_percentile is not None:
        filtered = filtered[filtered.scores>np.percentile(filtered.scores,top_percentile)]
        limit = len(filtered)
    else:
        raise ValueError("Specify either 'k' or 'percentile' - not both")

    top_n = filtered.index.tolist()[:limit+1]
    new_columns = [x for i,x in enumerate(features.columns) if i in top_n]
    return features[new_columns]

def preprocess_molecule_df(molecule_df,descriptor_df,y_col="Y",n=2,rng=5):
    """
    Function unique for this case, as it assumes
    that Y variable is logarithmic.
    The values logged in brackets can be ignored.
    """
    logging.info(f"Initializing activity sampler with the parameters: groups:{n}")

    _mols = list()
    for i,g in enumerate(ActivitySampler(molecule_df,descriptor_df,y_col=y_col,n=n,rng=rng).groups):
        _min = g[y_col].min()
        _max = g[y_col].max()

        _logmin = 10**(_min)
        _logmax = 10**(_max)
        if _min==0:
            _logmin=0
        if _max==0:
            _logmax=0

        logging.info(f"Group {i+1} between {_min:.4f} ({_logmin:.0f}) - {_max:.4f} ({_logmax:.0f}) of activity - consists of {len(g)} elements")
        _mols.append(g)
    molecule_df2 = pd.concat(_mols,axis=0)

    excluded = set(molecule_df.index) - set(molecule_df2.index)
    excluded = list(excluded)
    logging.info(f"Excluded {len(excluded)} compound(s) - {','.join(excluded)}")

    descriptor_df2 = descriptor_df.loc[molecule_df2.index]
    return molecule_df2, descriptor_df2

def score_function(feature_matrix,target,molecule_df,descriptor_df,model,groups=2,n_scores=5):
    "wrap around for partial"
    scorer = CrossValidator(molecule_df,descriptor_df,feature_matrix,target,groups=groups,n_scores=n_scores,model=model,)
    return scorer.score()

def purging_unfit_genes(features,target,y_col,
                        pop_size,start_genes,
                        scorer_func,parent_func,
                        min_fitness=0.5,genalg_params=None
                        ):
    """
    Retains only working genes
    """
    if genalg_params is None:
        genalg_params = dict()

    fit_individuals = list()

    while True:
        galg = GeneticAlgorithm(features,target[y_col],pop_size,start_genes,scorer_func,parent_func)
        galg.__dict__.update(genalg_params)
        galg.procreate()
        storage = galg.eugenic_storage

        fitness_features = [(key,[x for i,x in enumerate(features.columns) if vector[i]==1]) for key,vector in storage if key>min_fitness]
        if len(fitness_features)==0:
            logging.info("No more fit individuals")
            break
        logging.info(f"Found {len(fitness_features)} models with minimum fitness of {min(fitness_features)}")
        fit_individuals.extend(fitness_features)

        features_this_cycle = [x[1] for x in fitness_features]
        features_this_cycle = list({item for sublist in features_this_cycle for item in sublist})
        todrop = [x for x in features if x not in features_this_cycle]

        if len(todrop)==0:
            logging.info("No more unfit genes")
            break

        features = features.drop(todrop,axis=1)
        if features.shape[1]<start_genes:
            start_genes = features.shape[1]

        start_genes = start_genes-1
        logging.info(f"Dropped {len(todrop)} genes - {features.shape[1]} genes remaining - entering with {start_genes} genes active")
        if start_genes<1:
            logging.info("No more starting genes")
            break

    return sorted(fit_individuals,reverse=True)

def filter_vectors(vector_dict):
    """
    Does the following:
        - filters out same individuals with different fitness,
            -it takes the lowest fitness
        - calculates how often a gene is present in the population
    """
    #filtered_model = filter_vectors(result_models)
    vectors = [",".join(sorted(x[1])) for x in vector_dict] #vector_dict
    fitnesses = [x[0] for x in vector_dict] #vector_dict

    vector_df = pd.DataFrame([vectors,fitnesses],index=["V","F"]).T
    grouper = vector_df.groupby("V")["F"].min() #gets minimum value of Fitness
    grouper = grouper.sort_values(ascending=False)

    #convert grouper to dict
    value = grouper.index.tolist()
    key = grouper.tolist()
    grouper = list(zip(key,value))

    return grouper

def filter_models_via_params(models,model):
    """
    It's possible that two models are the same.
    For example, consider two models with 3 features:
        model_1 : A, B, C
        model_2 : A, B, D
    They differ in the last feature
    Upon estimating parameters, it happens that
    parameter for C and D is 0, so model_1 and model_2 are equivalent.
    This is something that can happen with models that implement feature selection,
    such as Lasso regularization
    """
    indices_lis = []
    parameters_lis = []
    with_coefs = []
    model_class = MODEL_DICTIONARY.get(model)
    for i,marko in enumerate(models):

        first = marko[1].split(",")

        first_feat = features[first]
        first_target = target.loc[first_feat.index]
        #model_class = MODEL_DICTIONARY.get(model)
        model_class.fit(features,first_target[1])

        coef_dict = dict(zip(first,model_class.coef_.flatten()))
        coef_dict = {x:y for x,y in coef_dict.items() if y!=0}

        if hasattr(model_class,"intercept_"):
            intercept = model_class.intercept_
        if hasattr(model_class,"y_mean_"):
            intercept = model_class.y_mean_[0]

        coef_dict["INTERCEPT"] = intercept

        if len(coef_dict)<2:
            continue
        beta_params = ",".join(sorted(list(coef_dict.keys())))
        if beta_params in parameters_lis:
            continue
        parameters_lis.append(beta_params)
        indices_lis.append(i)
        with_coefs.append(coef_dict)

    new_models = [x for i,x in enumerate(models) if i in indices_lis]

    vectors = [x.split(",") for x in parameters_lis]
    flattened_vectors = [item for sublist in vectors for item in sublist]
    set_vectors = set(flattened_vectors)
    #frequency of a descriptor in the model
    commonality = {x:flattened_vectors.count(x)/len(vectors) for x in set_vectors}

    return new_models, with_coefs, commonality

def predict_molecules(vector_dict,descriptor_df,molecule_df,random_molecules,y_col,model,fitness_limit=0.7):
    """
    Predicts molecules - extension function
    """
    predicted = list()
    model = MODEL_DICTIONARY.get(model)

    for fitness,vector_string in vector_dict:
        if fitness < fitness_limit:
            continue
        columns = vector_string.split(",")
        if len(columns)<2:
            continue

        features = descriptor_df[columns]
        features_random = random_molecules[columns]
        model.fit(features,molecule_df[y_col])
        y_predicted = model.predict(features_random).flatten()

        series = pd.Series(y_predicted,index=random_molecules.index,name=f"{fitness:.2f}")
        predicted.append(series)
    if len(predicted)==0:
        return None
    df = pd.concat(predicted,axis=1)
    return df

def test_compounds(input_df,y,alternative="greater",alpha=0.001,method="holm"):
    """
    Performs statistical testing on the dataframe, its one sided T Test
    """
    df = input_df.copy()
    logging.info(f"Statistics - pop_value:{y:.4f}  alpha:{alpha:.4f}  method:{method}  tails:{alternative}")
    pvalues = df.apply(lambda x:stats.ttest_1samp(x.values,popmean=y,alternative=alternative)[1],axis=1)
    adjusted = pg.multicomp(pvalues.values,method="holm")[1]
    adjusted = pd.Series(adjusted,index=pvalues.index)
    value = adjusted[adjusted<alpha].index.tolist()
    return value

#%% TRIPLE FUNCTIONS
def prefilter_the_features(features,target,y_col,
                           correlation_limit,repeated_value_limit,
                           activity_sampling_groups,rng,
                           alpha,top_percentile,k,extra_genes):
    """"
    Prefilters the features:
        - drops features correlated above 'correlation_limit'
        - drops features who have same value in 'repeated_value_limit' cases
        - splits features into activity sampling so the groups are even
        via 'activity_sampling_groups'
        - performs univariate testing to select best features
            ('top_percentile' or 'k', use 'alpha' for statistical significance)
        -adds extra features that aren't the best
    """
    features = drop_intercorrelated(features,corr=correlation_limit)
    features = drop_repeating_values(features,limit=repeated_value_limit)
    target, features = preprocess_molecule_df(target,features,y_col,n=activity_sampling_groups,rng=rng)

    top_features = select_top_features(features,target[y_col],top_percentile=top_percentile)

    other_features = pd.Series([x for x in features.columns if x not in top_features])
    other_features = other_features.sample(n=extra_genes,random_state=rng).tolist()

    columns = top_features.columns.tolist() + other_features
    features = features[columns]
    return features, target

def implement_model_selection(features,target,y_col,
                              model_name,groups,min_fitness,genalg_params):
    """
    As it states
    """
    model = MODEL_DICTIONARY.get(model_name)
    scorer_func = partial(score_function,molecule_df=target,descriptor_df=features,model=model,
                          groups=groups)

    start_genes = genalg_params.pop("start_gene")
    pop_size = genalg_params.pop("pop_size")
    parent_sel = genalg_params.pop("parent_sel")

    parent_func = PARENT_SELECTION[parent_sel]

    if start_genes is None:
       start_genes = features.shape[1] // 2
    if pop_size is None:
        pop_size = int(1.5 * features.shape[0])

    result_models = purging_unfit_genes(features,target,1,pop_size,start_genes,
                                        scorer_func,parent_func,
                                        genalg_params=genalg_params,min_fitness=min_fitness)
    filtered =  filter_vectors(result_models)
    filtered, parameters, common = filter_models_via_params(filtered,model_name)

    return filtered, parameters, common

def unify_models(models,features,target,y_col,random_molecules,
                 model,post_params):
    """
    unifies the models by implementing statistical testing

    """
    params = {k:v for k,v in post_params.items()}
    min_fitness = params.pop("min_fitness")


    predicted = predict_molecules(models,features,target,random_molecules,1,model,fitness_limit=min_fitness)

    molecules = test_compounds(predicted,**params)

    if len(molecules)==0:
        return pd.DataFrame()

    survived = predicted.loc[molecules]
    predicted_meanvec = survived.mean(axis=1).sort_values(ascending=False)
    survived = survived.loc[predicted_meanvec.index]
    return survived


#%%
if __name__=="__main__":

    features,target = prefilter_the_features(features,target,y_col=1,**pre_params)
    models, parameters, common = implement_model_selection(features,target,1,model,
                                               pre_params["activity_sampling_groups"],post_params["min_fitness"],
                                               genalg_params)

    model_dataframe = pd.DataFrame(parameters)
    model_dataframe.insert(0,"average_R_squared",[x[0] for x in models])

    predicted_molecules = unify_models(models,features,target,1,random_molecules,model,post_params)

    predicted_molecules.to_csv(output_predicted)
    model_dataframe.to_csv(output_models)


