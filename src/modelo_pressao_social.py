import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List
import plots

class ModeloPressaoSocial:

    def __init__(self, 
                n_actors:int, 
                influence_mode:str, 
                possible_opinions:List[int] = None, 
                beta = 1,
                U0 = None, # not necessary if possible_opinions and opinions_probabilities
                randomize_start_opinions:bool = False,
                bots_positions:List[int] = [],
                bots_opinion:int = None,
                
                ):
        
        self.actors_influencers = self.make_actors_influencers(
                                        n_actors = n_actors, 
                                        influence_mode = influence_mode,
                                        bots_positions = bots_positions
                                    )
        
        self.initial_pressures = self.make_initial_pressures(n_actors = n_actors )
        
        self.n_actors = n_actors
        self.influence_mode = influence_mode
        self.possible_opinions = possible_opinions
        self.randomize_start_opinions = randomize_start_opinions
        self.bots_positions = bots_positions
        self.bots_opinion = bots_opinion
        self.beta = beta
        
        

        return 
    
    def make_actors_influencers(self, 
                                n_actors, 
                                influence_mode,
                                bots_positions = None,
                                ):

        valid_influence_modes = ['circular_neighbors', 'all', 'linear_neighbors',
                                 'erdos_renyi', 'watts_strogatz']
        if influence_mode not in valid_influence_modes:
            raise ValueError("influence_mode must be one of %r." % valid_influence_modes)

             

        # creates a list of the actors positions, skipping bots
        actors_pos = []
        i = 1
        while len(actors_pos) < n_actors:
            if i not in bots_positions:
                actors_pos.append(i)
            i+=1


        last_actor = max(actors_pos + bots_positions)

        
        actors_influencers = dict()
        if influence_mode == 'circular_neighbors':
            
            actors_influencers = {f'actor_{n}':{'is_bot':False, 'influencers':[f'actor_{n-1}', f'actor_{n+1}']} for n in range(1,last_actor+1)}
            
            actors_influencers['actor_1'] = {
                'is_bot':False, 
                'influencers':[f'actor_{last_actor}', 'actor_2']
            }

            actors_influencers[f'actor_{last_actor}'] = {
                'is_bot':False,
                'influencers':[f'actor_{last_actor-1}', f'actor_1']
            }

        elif influence_mode == 'linear_neighbors':
            actors_influencers = {f'actor_{n}':{'is_bot': False,'influencers':[f'actor_{n-1}', f'actor_{n+1}']} for n in range(2,last_actor)}
            actors_influencers[f'actor_{last_actor}'] = {
                'is_bot': False,
                'influencers':[f'actor_{last_actor-1}']
            }
            actors_influencers['actor_1'] = {
                'is_bot': False,
                'influencers':[f'actor_2']
            }

        elif influence_mode == 'all':
            actors = range(1,last_actor+1)
            actors_influencers = {f'actor_{n}':{'is_bot':False,'influencers':[f'actor_{i}' for i in actors if i != n]} for n in actors}

        
        
        
        # insert bots
        if bots_positions != None:
            for bot_pos in bots_positions:
                actors_influencers[f'actor_{bot_pos}'] = {
                    
                    'is_bot': True,
                    'influencers':[]
                
                }
        # organize actors in dict
        actors_influencers = {f'actor_{pos}':actors_influencers[f'actor_{pos}'] for pos in range(1,last_actor+1)}

        return actors_influencers  
    
    def make_initial_pressures(self, n_actors):
        return {f'actor_{i+1}':0 for i in range(n_actors)}
    
    
    def pressure_function(self, Un, beta):
        return np.exp(beta*Un)
    
    
    def pi(self, Un, chosen_actor, chosen_opinion):
        
        Un_1 = Un + chosen_opinion
        Un_1[chosen_actor] = 0
        
        return Un_1
    
    def run(self, max_iterations:int):
        
        actors_influencers = self.actors_influencers
        Un = pd.Series(self.initial_pressures)
        
        
        self.run_info = {}
        for n in range(1, max_iterations+1):
            
            P_A_given_U = Un.apply(lambda Un_a: self.pressure_function(Un_a, self.beta) + self.pressure_function(Un_a, -self.beta))
            P_A_given_U = P_A_given_U/P_A_given_U.sum()
            
            chosen_actor = np.random.choice(P_A_given_U.index, 1, replace=True, p=P_A_given_U.values)[0]
            
            P_O_given_A_U = {opinion:self.pressure_function(Un[chosen_actor], opinion*self.beta) for opinion in self.possible_opinions}
            P_O_given_A_U = pd.Series(P_O_given_A_U)
            P_O_given_A_U = P_O_given_A_U/P_O_given_A_U.sum()
   
            chosen_opinion = np.random.choice(P_O_given_A_U.index, 1, replace=True, p=P_O_given_A_U.values)[0]
            
            Un = self.pi(Un, chosen_actor, chosen_opinion)
            
            
            self.run_info[n] = {
                
                    'chosen_actor': chosen_actor,
                    'chosen_opinion': chosen_opinion,
                    'Un': Un,
                    'P_A_given_U':P_A_given_U,
                    'P_O_given_A_U':P_O_given_A_U
                    
                }

            
        
        return self.run_info
    
    
    
    
    
    def make_graph(self):
        self.graph = plots.plot_actors_and_influencers(self.actors_influencers)
        return self.graph