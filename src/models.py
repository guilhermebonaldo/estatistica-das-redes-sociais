import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List
import plots

class ModeloVotante:

    def __init__(self, 
                n_actors:int, 
                influence_mode:str,
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
            possible inputs: ['circular_neighbors', 'all', 'linear_neighbors']
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
                                bots_positions = None,
                                ):

        valid_influence_modes = ['circular_neighbors', 'all', 'linear_neighbors']
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

    

    def run(self, max_iterations:int):
    
        # get dict of actors influencers
        actors_influencers = self.actors_influencers

        # make initial opinions list
        initial_opinions = self.initial_opinions = self.make_initial_opinions(
                                        n_actors = self.n_actors, 
                                        possible_opinions = self.possible_opinions, 
                                        opinions_probabilities = self.opinions_probabilities,
                                        bots_positions = self.bots_positions,
                                        bots_opinion = self.bots_opinion, 
                                        randomize = self.randomize_start_opinions
                                    )


        # make opinions dataframe (columns: actors, rows:rounds)
        X = pd.DataFrame(columns = actors_influencers.keys())
        # initial opinions on dataframe
        X.loc[0] = initial_opinions

        historical_influences = {}
        for t in range(1, max_iterations+1):

            # choose actor to emit opinion 
            chosen_actor = random.choice(X.columns)
            # get list of possible influencers
            possible_influencers = actors_influencers[chosen_actor]['influencers']

            if len(possible_influencers) > 0:
                # propagated opinions to next round
                X.loc[t] = X.iloc[-1]
                # choose influencer
                chosen_influencer = random.choice(possible_influencers)
                # new opinion
                new_opinion = X.loc[X.index[-1], chosen_influencer]
                # set new opinion
                X.loc[t, chosen_actor] = new_opinion

                historical_influences[t] = {
                    'influencer': chosen_influencer, 
                    'influenced': chosen_actor
                }
                

                # stops when converged
                if X.loc[t].nunique() == 1:
                    self.run_df = X
                    self.historical_influences = historical_influences
                    return (X, historical_influences)

            else: continue
            
            #print(historical_influences)

        self.run_df = X
        self.historical_influences = historical_influences
        
        return (X, historical_influences)


    def make_multiple_runs(self, n_runs, n_max_iter_per_run = 10000):
        
        results = {}
        for i in range(n_runs):
            X, influences = self.run(max_iterations = n_max_iter_per_run)
            results[i] = {
                'data': X,
                'influences': influences
            }
            
        return results


    def get_convergence_iteration(self, runs_list:List[pd.DataFrame]) -> List[int]:
        return [X.tail(1).index[0] for X in runs_list]


    def get_opinions_origins(self, run_df = None, historical_influences = None):

        if run_df is None:  
            run_df = self.run_df
        if historical_influences is None:  
            historical_influences = self.historical_influences

        # start with dataframe  of actor in columns and 
        # last opinion with the same value as the columns
        # the index must be the last iteration + 1
        # to be fullfiled by the process
        last_index = run_df.index.max()+1
        opinions_origins = pd.DataFrame(columns = run_df.columns, 
                                        data = [run_df.columns.tolist()], 
                                        index = [last_index] 
                                        )
        
        # get n rounds in reversed order
        rounds = sorted(historical_influences.keys())

        for i in reversed(rounds):
            opinions_origins.loc[i] = opinions_origins.loc[last_index]

            influenced = historical_influences[i]['influenced']
            influencer = historical_influences[i]['influencer']

            for actor in opinions_origins.columns:
                if opinions_origins.loc[i, actor] == influenced:
                    opinions_origins.loc[i, actor] = influencer
                    
            last_index = i


        self.opinions_origins = opinions_origins
        return opinions_origins
    
    def make_graph(self):
        self.graph = plots.plot_actors_and_influencers(self.actors_influencers)
        return self.graph