# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 16:10:32 2023

@author: Ivan
"""

import numpy as np
import pandas as pd
np.seterr(divide = 'ignore') 
rng = np.random.RandomState(0)


from scipy.spatial.distance import pdist, squareform
from functools import partial
from multiprocessing import Pool
import logging



#%% certain calculations

def calculate_gene_diversity(pop_vector):
    "average hamming distance"
    vector_df = pd.DataFrame(pop_vector)
    vector = pdist(vector_df,metric="hamming")
    dis_matrix = squareform(vector)
    weights = np.ones_like(dis_matrix,dtype=int)
    np.fill_diagonal(weights,0)
    average_distance = pd.Series(np.average(dis_matrix,weights=weights,axis=1),)
    return average_distance.mean()

def calculate_coefvar(fitness_array):
    "returns the coefficient of variation"
    return np.std(fitness_array)/np.mean(fitness_array)

def calculate_maxfitness(fitness_array):
    "returns the maximal fitness and its position"
    max_idx = np.argmax(fitness_array)
    max_fitness = fitness_array[max_idx]
    return max_fitness, max_idx
 

#%% parental selection functions
   
def parent_selection_rulette(population,population_fitness):
    """
    Selects parents using normalized population fitness as weights
    """
    #parents = list()
    #calculate normalized fitness
    norm_fitness = [x/sum(population_fitness) for x in population_fitness]
    #use it as weights to numpy random
    indices_population = np.arange(0,len(population),1)
    indices_parents = np.random.choice(indices_population,len(population),p=norm_fitness)
    parents = [population[x] for x in indices_parents]
    return parents

def parent_selection_rank(population,population_fitness,sp=2):
    """
    Selects parents using rank normalization. 
    I'm implementing Baker's formula for rank probability depending on 
    "sp" which stands for selection pressure:
        P(Ri) = 1/n [sp -(2sp-2)(i-1)/(n-1)]
    sp is from 1 to 2 with 2 indicating high selection pressure
    """
    #sort parents by fitness
    fitness_rank = np.argsort(population_fitness)[::-1]
    #sorted_fitness = np.argsort(population_fitness)
    sorted_population = [population[x] for x in fitness_rank]
    #get constants
    
    n = len(sorted_population)
    #rankify
    baker_form = [(sp/n) - (2*sp-2)*(i-1)/(n*n) for i in range(1,n+1)]
    baker_form = [x/sum(baker_form) for x in baker_form]
    #choose
    indices_population = np.arange(0,n,1)
    indices_parents = np.random.choice(indices_population,n,p=baker_form)
    parents = [sorted_population[x] for x in indices_parents]
    return parents

def parent_selection_steadystate(population,population_fitness,fitness_limit=0.25):
    """
    In this selection, individuals below certain fitness threshold are "killed",
    and the gene pool is replenished via procreation from survivors untill
    the same number of individuals comprises population.
    
    Two caveats:
        1) if fitness_limit is too soft, all will survive
        2) if fitness_limit is too hard, none will survive
    In both cases fitness_limit will be changed:
        1) fitness_limit will be increased
        2) fitness_limit will be descreased
    """
    
    pop_size = len(population)
    n_survivors = pop_size
    delta = abs(fitness_limit)/3
    
    if delta==0:
        delta = 0.1
    
    while n_survivors == pop_size:
        survived_idx = [i for i,x in enumerate(population_fitness) if x>=fitness_limit]
        survivors = [population[x] for x in survived_idx]
        n_survivors = len(survivors)
        #logging.info("allsurvive",pop_size,n_survivors,fitness_limit)
        fitness_limit += delta
    
    while n_survivors == 0:
        survived_idx = [i for i,x in enumerate(population_fitness) if x>=fitness_limit]
        survivors = [population[x] for x in survived_idx]
        n_survivors = len(survivors)
        #logging.info("nonsurvive",pop_size,n_survivors,fitness_limit)
        fitness_limit -= delta
    
    logging.info(f"\t{n_survivors} individuals out of {pop_size} survived threshold of {fitness_limit}")
    parents = survivors
    while len(parents)<pop_size:
        parents.extend(GeneticAlgorithm.mate_population(parents))
    else:
        parents = parents[:pop_size]
    return parents

#%%main function  
class GeneticAlgorithm():
    
    #limitation parameters
    cov_limit = 0.000005 # limit of coefficient of variation
    gen_limit = 300 # limit to the number of generations
    fitness_limit = 0.90 # limit of max fitness
    
    #mutation parameters
    mutation_probability = 0.001 #probability of a mutation event happening
    mutating_individuals = 0.8 #percentage of individuals mutating in a mutation event
    mutation_type = "inactivation" #mutation type
    
    #procreation parameters
    procreation_probability = 0.78 # probability of procreation
    monogamy = True #used for procreating 
    npoint = 5 #used for procreation - number of segments to do cross-over
    eugenic_generation_limit = 20 #every generation insert stored samples into population
    eugenic_mixture = 0.9 #percentage of eugenic that is drawn from population

    #calculation parameters
    n_cores = 1 #cores used for calculating 
    
    
    def __init__(self,data,target,pop_size,top_number,scorer_func=None,parent_func=None):
        self.data = data
        self.target = target
        self.gene_length = self.data.shape[1]
        self.pop_size = pop_size
        self.scorer_func = scorer_func
        self.select_parents = parent_func
        self.max_active_genes = top_number
        
        self.current_population = GeneticAlgorithm.initialize_population(self.gene_length,self.pop_size,top_number)
        self.previous_population = None
        self.previous_fitness = None
        self.generation = 0
        self.eugenic_storage = list()
        self.max_fitness = (0,0,0) #fitness | generation | bitvector
        
        if self.scorer_func is None or self.select_parents is None:
            raise KeyError("Must set scorer and parent selector functions")
                
    @staticmethod
    def initialize_population(c,population_size,top_number):
        """
        Creates a list of 'population_size' size of
        list of 'c' size that has random 0s and 1s
        and the number of 1s is equal to top number
        c : length of "chromosome" (the number of columns)
        population_size : the number of rows
        top_number : the amount of 1s to be in each individual
        
        """
        if top_number>=c:
            raise ValueError("top_number must be smaller than c")
        population = list()
        for i in range(population_size):
            individual_array = np.concatenate((np.ones(top_number,dtype=int), np.zeros(c-top_number,dtype=int)))
            np.random.shuffle(individual_array)
            population.append(list(individual_array))
        return population
    
    def calculate_fitness(self,population):
        data = self.data
        target = self.target
        scorer_func = self.scorer_func
        
        ##create multiple dataframes ("features")
        columnsall = [[i for i,x in enumerate(individual) if x== 1] for individual in population]
        features = [data.iloc[:,columns] for columns in columnsall]
        
        #create scorer function that takes in a dataframe then outputs a score
        fitness_function = partial(scorer_func,target=target)
        
        if self.n_cores == 1:
            return self.calculate_fitness_onethread(features,fitness_function)
        return self.calculate_fitness_mthreads(features,fitness_function)
            
    @staticmethod
    def calculate_fitness_onethread(features,func):
        return [func(feature_matrix=f) for f in features]
    
    def calculate_fitness_mthreads(self,features,func):
        with Pool(self.n_cores) as pool:
            result = pool.map(func, features)
            _result = [r for r in result]
        return _result
    
    def iteration(self):
        
        if self.generation % self.eugenic_generation_limit == 0 and self.generation!=0:
            num_eugenics = int(self.pop_size*self.eugenic_mixture)
            num_current = self.pop_size - num_eugenics
            #logging.info(num_eugenics,num_current,num_eugenics+num_current,self.pop_size)
            eugenic_population = [x[1] for x in self.eugenic_storage[:num_eugenics]]
            eugenic_fitness = [x[0] for x in self.eugenic_storage[:num_eugenics]]
            logging.info(f"Eugenic injection! - Added {len(eugenic_population)} to current population")
            #logging.info(len(eugenic_population))
            population = self.current_population[:num_current]
            fitness = self.calculate_fitness(population)

            population.extend(eugenic_population)
            fitness.extend(eugenic_fitness)

        else:
            population = self.current_population
            fitness = self.calculate_fitness(population) #TODO -here
            #store some delicous eugenics
            self.extend_eugenic_storage(population,fitness)
        
        parents = self.select_parents(population,fitness)
        next_generation = self.mate_population(parents,self.procreation_probability,self.npoint,self.monogamy)
        
        #add mutation
        if np.random.uniform()<=self.mutation_probability:
            self.mutate_population(next_generation, self.mutation_type,self.mutating_individuals)
        
        self.previous_population = population
        self.previous_fitness = fitness
        self.current_population = next_generation
        
        coefvar = calculate_coefvar(fitness)
        max_fit,max_idx = calculate_maxfitness(fitness)
        return coefvar,max_fit,max_idx
        
    def extend_eugenic_storage(self,population,fitness):
        eugenics = zip(fitness,population)
        self.eugenic_storage.extend(eugenics)
        self.eugenic_storage = sorted(self.eugenic_storage,reverse=True)
        
        #self.eugenic_storage = sorted(self.eugenic_storage)[::-1]
        if len(self.eugenic_storage)>self.pop_size:
            self.eugenic_storage = self.eugenic_storage[:self.pop_size]
            
    def calculate_final_statistics(self):
        """
        Calculates the following values for the top 15 models:
            For average fitness:
                Minimum, Maximum, Mean of fitness 
            For vector:
                Hamming distance
        Hamming distance tells how different the vectors are and goes from 0 to 1
        0 means totally the same, 1 means totally different
        """
        
        storage = self.eugenic_storage[:15]
        fitness = np.array([x[0] for x in storage])
        vectors = [x[1] for x in storage]
        
        logging.info(f"FITNESS_STATISTICS : MIN:{fitness.min():.2f}  MAX:{fitness.max():.2f} AVG:{fitness.mean():.2f}")
    
        vector_df = pd.DataFrame(vectors)
    
        vector = pdist(vector_df,metric="hamming")
        dis_matrix = squareform(vector)
        weights = np.ones_like(dis_matrix,dtype=int)
        np.fill_diagonal(weights,0)
        average_distance = pd.Series(np.average(dis_matrix,weights=weights,axis=1),)
    
        logging.info(f"VECTOR_STATISTICS: MIN:{average_distance.min():.2f}  MAX:{average_distance.max():.2f} AVG:{average_distance.mean():.2f}")
        return dict(zip(fitness,vectors))
        
    def procreate(self):
        if self.scorer_func is None:
            raise ValueError("Scorer function must be set")
        #create children and start iteration
        #prev_cov, prev_fit, prev_man = self.iteration()
        self.iteration()
        
        self.generation += 1
        #logging.info(self.generation,prev_cov,prev_fit,prev_man)
        #self.log.append([self.generation,prev_cov,prev_fit,self.previous_population[prev_man]])
        
        while True:
            #cur_cov, cur_fit, cur_man = self.iteration()
            self.iteration()
            self.generation += 1
            
            #statistics
            fitness_vector = self.previous_fitness 
            population_vector = self.previous_population
            
            
            max_fitness, idx = calculate_maxfitness(fitness_vector)
            avg_fitness = sum(fitness_vector)/len(fitness_vector)
            #fitest_ind = population_vector[idx]
            max_fitness = max(fitness_vector)
            
            #store maximum fitness
            generation = self.generation-1
            maximum_fitness = (max_fitness,generation,population_vector[idx])
            if maximum_fitness[0]>self.max_fitness[0]:
                self.max_fitness = maximum_fitness
            
            
            delta_gen = maximum_fitness[1] - self.max_fitness[1]
            max_fitness = self.max_fitness[0]
            genes_active = sum([sum(x) for x in population_vector])/self.pop_size
            gene_diversity = calculate_gene_diversity(population_vector)
            
            #statistics
            #gen - generation
            #max_fitness - maximum fitness of generation
            #avg_fitness - average fitness of generation
            #fittest - maximum fitness ever achieveved
            #delta_g - how many generations ago
            #genes_active - average amount of active genes
            
            statistics = f"GEN:{generation}  AVG_FIT:{avg_fitness:.2f}  MAX_FIT:{max_fitness:.2f}  "
            statistics = statistics + f"FITTEST:{max_fitness:.2f}  DELTA_G:{delta_gen}  AVERAGE_ACTIVE_GENES:{genes_active:.1f}  "
            statistics = statistics + f"AVERAGE_GENE_DIVERSITY:{gene_diversity:.4f}"
            logging.info(statistics)
            #change the mutation type based on genes_active
            if genes_active<=0.2*self.max_active_genes:
                self.mutation_type == "activation"
            if genes_active>=0.6*self.max_active_genes:
                self.mutation_type == "inactivation"
           
            if max_fitness>=self.fitness_limit:
                logging.info("fitlim")
                break
            if self.generation>=self.gen_limit:
                logging.info("genlim")
                break
                    
        #fitness_value, _, fitness_vector = self.max_fitness
        #columns_idx = [i for i,x in enumerate(fitness_vector) if x== 1]
        #final_features = self.data.iloc[:,columns_idx]
        return self.calculate_final_statistics()
        #return(fitness_value, final_features)
        
    @staticmethod
    def crossover_Npoint(father,mother,npoint=3):
        """
        Performs "mixture" of genetic material of 'father'
        and 'mother'.
        Father and mother should be a list of equal size consisting
        of 0 and 1s. Specifying points segments the list in N+1 segments.
        Example, for npoint = 2 we get the segment:
        
        >>> 00 | 1001 | 0111
        >>> 10 | 1100 | 0011
        
        Then those segments are flipped over to produce two offsprings
        
        >> 00 | 1100 | 0011
        >> 10 | 1001 | 0111
        
        Where points are made is randomized
        The return 
        
        """
        c = len(father) #len father and mother should be the same
        points = np.sort(np.random.choice(np.arange(0,c,1),npoint,replace=False),)
        
        child1 = father[:points[0]]
        child2 = mother[:points[0]]

        for start,end in zip(points[:-1],points[1:]):
            father,mother = mother,father
            child1.extend(father[start:end])
            child2.extend(mother[start:end])
        else:
            child1.extend(mother[end:])
            child2.extend(father[end:])
        return child1, child2

    @staticmethod
    def mate_population(potential_parents,probability=0.78,npoint=4,monogamy=True):
        """
        Function for creating the next generation.
        It takes a pool of 'potential_parents' which is a list of lists, for example:
            >>> [
            >>> [1 1 1 0 0 1]
            >>> [1 0 0 1 1 0]
            >>> [1 1 0 1 0 0]
            >>> ]
        From this pool, two parents are selected with specified 'probability'
        and then 'mated' according to "crossover_Npoint" function to 
        create a new bit list
        
        Then one parent is removed from the iteration, and if 'monogamy' is specified
        the second parent is removed too. 
        The result is the next generation which consists of "children" and 
        all parents that did not procreate
        """
        
        
        pop_size = len(potential_parents)
        parent_index = list(range(pop_size))
        pairs = int((probability*pop_size)//2)

        procreated = list()
        next_generation = list()

        for i in range(pairs):
            parent1_idx, parent2_idx = np.random.choice(parent_index,2,replace=False)
            parent_index.remove(parent2_idx)
            if monogamy:
                parent_index.remove(parent1_idx)
            procreated.extend([parent1_idx,parent2_idx])
            father = potential_parents[parent1_idx]
            mother = potential_parents[parent2_idx]
            next_generation.extend(GeneticAlgorithm.crossover_Npoint(father,mother,npoint))

        spinsters = [x for i,x in enumerate(potential_parents) if i not in procreated]
        next_generation.extend(spinsters)
        return next_generation
        
    @staticmethod
    def mutate_population(population,mutation_type,individual_mutation=1):
        """
        Inplace mutation of population
        mutation_type:
            "activation" - 0 becomes 1
            "inactivation" - 1 becomes 0
            "swap" - 0 becomes 1 and 1 becomes 0
            "mix" - random selection between the two
        """    
        if mutation_type == "mix":
            mutation_type = np.random.choice(["activation","inactivation","swap"])

        mutated_count = 0
        for individual in population:
            if np.random.uniform()>individual_mutation:
                continue
            
            if mutation_type in ["inactivation","swap"]:
                index_1 = [i for i,x in enumerate(individual) if x==1]
                if not index_1:
                    continue
                choice_1 = np.random.choice(index_1)
                individual[choice_1]=0
            if mutation_type in ["activation","swap"]:
                index_0 = [i for i,x in enumerate(individual) if x==0]
                if not index_0:
                    continue
                choice_0 = np.random.choice(index_0)
                individual[choice_0]=1
            mutated_count += 1
        logging.info(f"mutated {mutated_count} individuals with {mutation_type} mutation")
