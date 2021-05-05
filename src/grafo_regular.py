import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List
import plots

class GrafoRegular:

    def __init__(self, 
                n_actors:int, 
                influence_mode:str, # erdos_renyi, watts_strogatz
                connection_probability:float,
                possible_opinions:List[int] = None, 
                opinions_probabilities:List[float] = None, 
                initial_opinions_list = None, # not necessary if possible_opinions and opinions_probabilities
                randomize_start_opinions:bool = False,
                bots_positions:List[int] = [],
                bots_opinion:int = None,
                
                ):

        """
        inits ModeloVotante
        
        Parameters
        ----------
        n_actors:int
            nuber of actors in the network
        influence_mode:str
            how influences are assigned
            possible inputs: ['regular']
        possible_opinions:List[int]
            list of possible opinions
            e.g.: [-1,1]
        opinions_probabilities:List[float]
            probabilities of opinions setted in possible_opinions
        initial_opinions_list: List[int]
            [optional]
            custom list of opinions
            if passed, possible_opinions and opinions_probabilities will be ignored
        
        randomize_start_opinions: bool
            if True, opinions list will be shuffled
        
        bots_positions: List[int]
            list of robots positions in the network
        
        bots_opinion: int
            bots opinion
        Returns
        -------
 
        """
        
        # checks
        if possible_opinions == None or opinions_probabilities == None: 
            assert initial_opinions_list != None , \
            'If initial_opinions_list is not specified, \
                possible_opinions and opinions_probabilities are necessary!'
    
        if initial_opinions_list:
            assert len(initial_opinions_list) == n_actors, \
                    'number of opinions is different from n_actors'

        if bots_positions:
            assert bots_opinion, \
                    'When bots exists its needed to specify an opinion'


        self.actors_influencers = self.make_actors_influencers(
                                        n_actors = n_actors, 
                                        influence_mode = influence_mode,
                                        connection_probability = connection_probability,
                                        bots_positions = bots_positions
                                    )
                                    
        if initial_opinions_list == None:
            self.initial_opinions = self.make_initial_opinions(
                                        n_actors = n_actors, 
                                        possible_opinions = possible_opinions, 
                                        opinions_probabilities = opinions_probabilities,
                                        bots_positions = bots_positions,
                                        bots_opinion = bots_opinion, 
                                        randomize = randomize_start_opinions
                                    )

        elif initial_opinions_list: 
            self.initial_opinions = initial_opinions_list


        
        self.n_actors = n_actors
        self.influence_mode = influence_mode
        self.possible_opinions = possible_opinions
        self.opinions_probabilities = opinions_probabilities
        self.randomize_start_opinions = randomize_start_opinions
        self.bots_positions = bots_positions
        self.bots_opinion = bots_opinion

        return 


    def make_actors_influencers(self, 
                                n_actors, 
                                influence_mode,
                                connection_probability,
                                bots_positions = None,
                                ):

        valid_influence_modes = ['erdos_renyi','regular']
        if influence_mode not in valid_influence_modes:
            raise ValueError("influence_mode must be one of %r." % valid_influence_modes)

             

        
        actors_influencers = pd.DataFrame(columns = [f'actor_{i}' for i in range(1,n_actors+1)],
                                          index = [f'actor_{i}' for i in range(1,n_actors+1)]
                                          )
        if influence_mode == 'erdos_renyi':
            
            for v in range(n_actors):
                for v_ in range(v+1, n_actors):
                    
                    if random.uniform(0, 1) <=  connection_probability:
                        actors_influencers.iloc[v_,v] = 1
                    else:
                        actors_influencers.iloc[v_,v] = 0
                     
                
           

        actors_influencers = actors_influencers.fillna(0)
        actors_influencers = actors_influencers.add(actors_influencers.T)
        
        return actors_influencers  


    def make_initial_opinions(self, 
                              n_actors, 
                              possible_opinions, 
                              opinions_probabilities,
                              bots_positions = None,
                              bots_opinion = None, 
                              randomize = False
                              ):
    
    
        initial_opinions = [[possible_opinions[i]]*int(n_actors*opinions_probabilities[i]) for i in range(len(possible_opinions))]
        # flatten list
        initial_opinions = [i for sublist in initial_opinions for i in sublist]

        if randomize:
            initial_opinions = random.sample(initial_opinions, len(initial_opinions))
            
        if bots_positions:
            for bot_pos in bots_positions:
                initial_opinions.insert(bot_pos-1, bots_opinion)
        

        return initial_opinions

    

    
    def D(self):       
        return self.actors_influencers.sum()
    
    def make_graph(self):
        self.graph = plots.plot_actors_and_influencers(self.actors_influencers)
        return self.graph