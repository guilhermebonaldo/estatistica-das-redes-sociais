import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import graphviz as gr


def plot_actors_and_influencers(actors_influencers):
    '''
    dot - “hierarchical” or layered drawings of directed graphs. 
           This is the default tool to use if edges have directionality.
    neato - “spring model” layouts. This is the default tool to use if the graph is not too large (about 100 nodes)
            and you don't know anything else about it. Neato attempts to minimize a global energy function, 
            which is equivalent to statistical multi-dimensional scaling.
    fdp - “spring model” layouts similar to those of neato, 
            but does this by reducing forces rather than working with energy.
    sfdp - multiscale version of fdp for the layout of large graphs.
    twopi - radial layouts, after Graham Wills 97. 
            Nodes are placed on concentric circles depending their distance from a given root node.
    circo - circular layout, after Six and Tollis 99, Kauffman and Wiese 02. 
            This is suitable for certain diagrams of multiple cyclic structures, 
            such as certain telecommunications networks.
    
    https://graphviz.org/doc/info/attrs.html
    '''
    
    g = gr.Digraph(
        
        'pipeline_graph', 
        engine = 'circo', 
        
        node_attr={
            'color': '#FFBE33',#'#FFA136',
            'style': 'filled', 
            'shape':'Mrecord',
            'fontname': 'Arial', 
            'fontsize':'10'
         }, 
        #filename='unix.gv',
        #strict = True
    )
    
    for actor in actors_influencers.keys():
        if actors_influencers[actor]['is_bot']:
            g.node(actor, color= '#FF0000')
            
        if actors_influencers[actor]['influencers']:
            for influencer in actors_influencers[actor]['influencers']:
                g.edge(
                    tail_name = influencer, 
                    head_name = actor, 
                    color = '#C0C0C0'
                )
                
                
    
                
    return g
