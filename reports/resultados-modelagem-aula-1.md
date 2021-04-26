```python
import sys
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns

%load_ext autoreload
%autoreload 2

sys.path.insert(1, '../src/')
pd.set_option("display.max_columns", 999)

import models
import plots
import utils
```


```python
def make_final_opinions_analysis(model):
    n_runs = 1000
    print(f'Numero de runs: {n_runs}')
    
    data = model.make_multiple_runs(n_runs = n_runs, n_max_iter_per_run = 10_000)
    final_opinions = pd.concat([data[run]['data'].tail(1).reset_index() for run in data.keys()], ignore_index = True)
    

    convergence = final_opinions[['actor_1', 'index']]

    convergence = convergence.groupby('actor_1').agg({'index':['count', 'mean', 'median', 'std']})
    convergence.columns = convergence.columns.droplevel()
    convergence.index = convergence.index.rename('opiniao')
    summary = convergence.copy()
    
    summary['opinions_prob'] = model.opinions_probabilities
    summary['random'] = model.randomize_start_opinions
    summary['influence_mode'] = model.influence_mode
    
    print('\n',summary,'\n')
    
    for i in [1,-1]:
        convergence = final_opinions[final_opinions.actor_1 == i].iloc[:, 0]        
        sns.kdeplot(convergence)
        plt.legend(labels=['1','-1'])

    plt.show()

    for i in [1,-1]:
        sns.displot(final_opinions[final_opinions.actor_1 == i].iloc[:, 0], bins = 20)
        plt.legend(labels=[i])
        plt.show()
        
    return data, summary.reset_index()
```

# Modelo Votante

O objetivo desse estudo é observar o tempo de convergencia de redes com 10 atores para as combinações das seguintes variáveis
- Desenho do grafo (3 categorias: fila, circular, todos influenciam todos)
- probabilidade para cada opinião inicial (-1, 1): (0.5, 0.5) e (0.7, 0.3)
- alocação das opiniões: aleatória ou separada


--------------------------------------

## atores em fila
## probabilidades das opiniões iniciais: (50%, 50%)
## alocação aleatória


```python
model = models.ModeloVotante(n_actors = 10,
                              influence_mode = 'linear_neighbors',
                              possible_opinions = [-1,1], 
                              opinions_probabilities = [5/10,5/10],
                              randomize_start_opinions= True,
                    
                              )

model.make_graph()
```




    
![svg](output_6_0.svg)
    




```python
data_1 = make_final_opinions_analysis(model)
```

    Numero de runs: 1000
    
              count        mean  median         std  opinions_prob  random  \
    opiniao                                                                 
    -1         509  164.836935     105  173.298440            0.5    True   
     1         491  144.181263     101  135.067291            0.5    True   
    
               influence_mode  
    opiniao                    
    -1       linear_neighbors  
     1       linear_neighbors   
    



    
![png](output_7_1.png)
    



    
![png](output_7_2.png)
    



    
![png](output_7_3.png)
    


-------------

## atores em fila
## probabilidades das opiniões iniciais: (50%, 50%)
## alocação separada


```python
model = models.ModeloVotante(n_actors = 10,
                              influence_mode = 'linear_neighbors',
                              possible_opinions = [-1,1], 
                              opinions_probabilities = [5/10,5/10],
                              randomize_start_opinions= False,
                              )

model.make_graph()
```




    
![svg](output_10_0.svg)
    




```python
data_2 = make_final_opinions_analysis(model)
```

    Numero de runs: 1000
    
              count        mean  median         std  opinions_prob  random  \
    opiniao                                                                 
    -1         503  206.248509     163  161.835647            0.5   False   
     1         497  198.372233     142  168.581619            0.5   False   
    
               influence_mode  
    opiniao                    
    -1       linear_neighbors  
     1       linear_neighbors   
    



    
![png](output_11_1.png)
    



    
![png](output_11_2.png)
    



    
![png](output_11_3.png)
    


------------------

## atores em fila
## probabilidades das opiniões iniciais: (70%, 30%)
## alocação aleatória


```python
model = models.ModeloVotante(n_actors = 10,
                              influence_mode = 'linear_neighbors',
                              possible_opinions = [-1,1], 
                              opinions_probabilities = [7/10,3/10],
                              randomize_start_opinions= True,
                              )

model.make_graph()
```




    
![svg](output_14_0.svg)
    




```python
data_3 = make_final_opinions_analysis(model)
```

    Numero de runs: 1000
    
              count        mean  median         std  opinions_prob  random  \
    opiniao                                                                 
    -1         711   99.457103      44  134.828981            0.7    True   
     1         289  194.543253     149  155.471703            0.3    True   
    
               influence_mode  
    opiniao                    
    -1       linear_neighbors  
     1       linear_neighbors   
    



    
![png](output_15_1.png)
    



    
![png](output_15_2.png)
    



    
![png](output_15_3.png)
    


------------------

## atores em fila
## probabilidades das opiniões iniciais: (70%, 30%)
## alocação separada


```python
model = models.ModeloVotante(n_actors = 10,
                              influence_mode = 'linear_neighbors',
                              possible_opinions = [-1,1], 
                              opinions_probabilities = [7/10,3/10],
                              randomize_start_opinions= False,
                              )

model.make_graph()
```




    
![svg](output_18_0.svg)
    




```python
data_4 = make_final_opinions_analysis(model)
```

    Numero de runs: 1000
    
              count        mean  median         std  opinions_prob  random  \
    opiniao                                                                 
    -1         738  138.127371    80.0  153.929957            0.7   False   
     1         262  268.610687   204.5  199.559191            0.3   False   
    
               influence_mode  
    opiniao                    
    -1       linear_neighbors  
     1       linear_neighbors   
    



    
![png](output_19_1.png)
    



    
![png](output_19_2.png)
    



    
![png](output_19_3.png)
    


------------------

## atores em circulo
## probabilidades das opiniões iniciais: (50%, 50%)
## alocação aleatória


```python
model = models.ModeloVotante(n_actors = 10,
                              influence_mode = 'circular_neighbors',
                              possible_opinions = [-1,1], 
                              opinions_probabilities = [5/10,5/10],
                              randomize_start_opinions= True,
                              #initial_opinions_list = [-1,-1,-1,-1,-1,1,1,1,1,1]
                              )

model.make_graph()
```




    
![svg](output_22_0.svg)
    




```python
data_5 = make_final_opinions_analysis(model)
```

    Numero de runs: 1000
    
              count        mean  median        std  opinions_prob  random  \
    opiniao                                                                
    -1         483  110.942029      84  92.960086            0.5    True   
     1         517  106.367505      80  94.038506            0.5    True   
    
                 influence_mode  
    opiniao                      
    -1       circular_neighbors  
     1       circular_neighbors   
    



    
![png](output_23_1.png)
    



    
![png](output_23_2.png)
    



    
![png](output_23_3.png)
    


-------------

## atores em circulo
## probabilidades das opiniões iniciais: (50%, 50%)
## alocação separada


```python
model = models.ModeloVotante(n_actors = 10,
                              influence_mode = 'circular_neighbors',
                              possible_opinions = [-1,1], 
                              opinions_probabilities = [5/10,5/10],
                              randomize_start_opinions= False,
                              )

model.make_graph()
```




    
![svg](output_26_0.svg)
    




```python
data_6 = make_final_opinions_analysis(model)
```

    Numero de runs: 1000
    
              count        mean  median        std  opinions_prob  random  \
    opiniao                                                                
    -1         495  116.006061      88  94.079205            0.5   False   
     1         505  122.562376      98  94.206969            0.5   False   
    
                 influence_mode  
    opiniao                      
    -1       circular_neighbors  
     1       circular_neighbors   
    



    
![png](output_27_1.png)
    



    
![png](output_27_2.png)
    



    
![png](output_27_3.png)
    


------------------

## atores em circulo
## probabilidades das opiniões iniciais: (70%, 30%)
## alocação aleatória


```python
model = models.ModeloVotante(n_actors = 10,
                              influence_mode = 'circular_neighbors',
                              possible_opinions = [-1,1], 
                              opinions_probabilities = [7/10,3/10],
                              randomize_start_opinions= True,
                              )

model.make_graph()
```




    
![svg](output_30_0.svg)
    




```python
data_7 = make_final_opinions_analysis(model)
```

    Numero de runs: 1000
    
              count        mean  median         std  opinions_prob  random  \
    opiniao                                                                 
    -1         680   75.977941    41.5   94.338786            0.7    True   
     1         320  137.471875   110.5  100.269195            0.3    True   
    
                 influence_mode  
    opiniao                      
    -1       circular_neighbors  
     1       circular_neighbors   
    



    
![png](output_31_1.png)
    



    
![png](output_31_2.png)
    



    
![png](output_31_3.png)
    


------------------

## atores em circulo
## probabilidades das opiniões iniciais: (70%, 30%)
## alocação separada


```python
model = models.ModeloVotante(n_actors = 10,
                              influence_mode = 'circular_neighbors',
                              possible_opinions = [-1,1], 
                              opinions_probabilities = [7/10,3/10],
                              randomize_start_opinions= False,
                              )

model.make_graph()
```




    
![svg](output_34_0.svg)
    




```python
data_8 = make_final_opinions_analysis(model)
```

    Numero de runs: 1000
    
              count        mean  median         std  opinions_prob  random  \
    opiniao                                                                 
    -1         701   91.649073      52  101.116296            0.7   False   
     1         299  148.903010     118  108.384324            0.3   False   
    
                 influence_mode  
    opiniao                      
    -1       circular_neighbors  
     1       circular_neighbors   
    



    
![png](output_35_1.png)
    



    
![png](output_35_2.png)
    



    
![png](output_35_3.png)
    


## todos influenciam todos
## probabilidades das opiniões iniciais: (50%, 50%)
## alocação aleatória


```python
model = models.ModeloVotante(n_actors = 10,
                              influence_mode = 'all',
                              possible_opinions = [-1,1], 
                              opinions_probabilities = [5/10,5/10],
                              randomize_start_opinions= True,
                              #initial_opinions_list = [-1,-1,-1,-1,-1,1,1,1,1,1]
                              )

model.make_graph()
```




    
![svg](output_37_0.svg)
    




```python
data_9 = make_final_opinions_analysis(model)
```

    Numero de runs: 1000
    
              count       mean  median        std  opinions_prob  random  \
    opiniao                                                               
    -1         504  56.099206      43  45.719193            0.5    True   
     1         496  62.945565      45  54.171101            0.5    True   
    
            influence_mode  
    opiniao                 
    -1                 all  
     1                 all   
    



    
![png](output_38_1.png)
    



    
![png](output_38_2.png)
    



    
![png](output_38_3.png)
    


-------------

## todos influenciam todos
## probabilidades das opiniões iniciais: (50%, 50%)
## alocação separada


```python
model = models.ModeloVotante(n_actors = 10,
                              influence_mode = 'all',
                              possible_opinions = [-1,1], 
                              opinions_probabilities = [5/10,5/10],
                              randomize_start_opinions= False,
                              )

model.make_graph()
```




    
![svg](output_41_0.svg)
    




```python
data_10 = make_final_opinions_analysis(model)
```

    Numero de runs: 1000
    
              count       mean  median        std  opinions_prob  random  \
    opiniao                                                               
    -1         509  56.960707      43  46.795526            0.5   False   
     1         491  57.723014      44  45.033464            0.5   False   
    
            influence_mode  
    opiniao                 
    -1                 all  
     1                 all   
    



    
![png](output_42_1.png)
    



    
![png](output_42_2.png)
    



    
![png](output_42_3.png)
    


------------------

## todos influenciam todos
## probabilidades das opiniões iniciais: (70%, 30%)
## alocação aleatória


```python
model = models.ModeloVotante(n_actors = 10,
                              influence_mode = 'all',
                              possible_opinions = [-1,1], 
                              opinions_probabilities = [7/10,3/10],
                              randomize_start_opinions= True,
                              )

model.make_graph()
```




    
![svg](output_45_0.svg)
    




```python
data_11 = make_final_opinions_analysis(model)
```

    Numero de runs: 1000
    
              count       mean  median        std  opinions_prob  random  \
    opiniao                                                               
    -1         702  43.548433      29  41.695518            0.7    True   
     1         298  71.684564      56  50.425942            0.3    True   
    
            influence_mode  
    opiniao                 
    -1                 all  
     1                 all   
    



    
![png](output_46_1.png)
    



    
![png](output_46_2.png)
    



    
![png](output_46_3.png)
    


------------------

## todos influenciam todos
## probabilidades das opiniões iniciais: (70%, 30%)
## alocação separada


```python
model = models.ModeloVotante(n_actors = 10,
                              influence_mode = 'all',
                              possible_opinions = [-1,1], 
                              opinions_probabilities = [7/10,3/10],
                              randomize_start_opinions= False,
                              )

model.make_graph()
```




    
![svg](output_49_0.svg)
    




```python
data_12 = make_final_opinions_analysis(model)
```

    Numero de runs: 1000
    
              count       mean  median        std  opinions_prob  random  \
    opiniao                                                               
    -1         720  43.912500    29.0  42.787593            0.7   False   
     1         280  71.264286    58.5  47.133473            0.3   False   
    
            influence_mode  
    opiniao                 
    -1                 all  
     1                 all   
    



    
![png](output_50_1.png)
    



    
![png](output_50_2.png)
    



    
![png](output_50_3.png)
    



```python

```


```python
results = [data_1[1],
            data_2[1],
            data_3[1],
            data_4[1],
            data_5[1],
            data_6[1],
            data_7[1],
            data_8[1],
            data_9[1],
            data_10[1],
            data_11[1],
            data_12[1]
          ]


df = pd.concat(results)

df.to_csv('../data/results/aula1.csv')
```


```python
df = pd.read_csv('../data/results/aula1.csv').iloc[:,1:]
```


```python
df[df.opinions_prob == 0.5].sort_values('mean')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>opiniao</th>
      <th>count</th>
      <th>mean</th>
      <th>median</th>
      <th>std</th>
      <th>opinions_prob</th>
      <th>random</th>
      <th>influence_mode</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-1</td>
      <td>504</td>
      <td>56.099206</td>
      <td>43.0</td>
      <td>45.719193</td>
      <td>0.5</td>
      <td>True</td>
      <td>all</td>
    </tr>
    <tr>
      <th>0</th>
      <td>-1</td>
      <td>509</td>
      <td>56.960707</td>
      <td>43.0</td>
      <td>46.795526</td>
      <td>0.5</td>
      <td>False</td>
      <td>all</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>491</td>
      <td>57.723014</td>
      <td>44.0</td>
      <td>45.033464</td>
      <td>0.5</td>
      <td>False</td>
      <td>all</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>496</td>
      <td>62.945565</td>
      <td>45.0</td>
      <td>54.171101</td>
      <td>0.5</td>
      <td>True</td>
      <td>all</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>517</td>
      <td>106.367505</td>
      <td>80.0</td>
      <td>94.038506</td>
      <td>0.5</td>
      <td>True</td>
      <td>circular_neighbors</td>
    </tr>
    <tr>
      <th>0</th>
      <td>-1</td>
      <td>483</td>
      <td>110.942029</td>
      <td>84.0</td>
      <td>92.960086</td>
      <td>0.5</td>
      <td>True</td>
      <td>circular_neighbors</td>
    </tr>
    <tr>
      <th>0</th>
      <td>-1</td>
      <td>495</td>
      <td>116.006061</td>
      <td>88.0</td>
      <td>94.079205</td>
      <td>0.5</td>
      <td>False</td>
      <td>circular_neighbors</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>505</td>
      <td>122.562376</td>
      <td>98.0</td>
      <td>94.206969</td>
      <td>0.5</td>
      <td>False</td>
      <td>circular_neighbors</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>491</td>
      <td>144.181263</td>
      <td>101.0</td>
      <td>135.067291</td>
      <td>0.5</td>
      <td>True</td>
      <td>linear_neighbors</td>
    </tr>
    <tr>
      <th>0</th>
      <td>-1</td>
      <td>509</td>
      <td>164.836935</td>
      <td>105.0</td>
      <td>173.298440</td>
      <td>0.5</td>
      <td>True</td>
      <td>linear_neighbors</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>497</td>
      <td>198.372233</td>
      <td>142.0</td>
      <td>168.581619</td>
      <td>0.5</td>
      <td>False</td>
      <td>linear_neighbors</td>
    </tr>
    <tr>
      <th>0</th>
      <td>-1</td>
      <td>503</td>
      <td>206.248509</td>
      <td>163.0</td>
      <td>161.835647</td>
      <td>0.5</td>
      <td>False</td>
      <td>linear_neighbors</td>
    </tr>
  </tbody>
</table>
</div>




```python
df[df.opinions_prob.isin([0.3,0.7])].sort_values(['mean'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>opiniao</th>
      <th>count</th>
      <th>mean</th>
      <th>median</th>
      <th>std</th>
      <th>opinions_prob</th>
      <th>random</th>
      <th>influence_mode</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-1</td>
      <td>702</td>
      <td>43.548433</td>
      <td>29.0</td>
      <td>41.695518</td>
      <td>0.7</td>
      <td>True</td>
      <td>all</td>
    </tr>
    <tr>
      <th>0</th>
      <td>-1</td>
      <td>720</td>
      <td>43.912500</td>
      <td>29.0</td>
      <td>42.787593</td>
      <td>0.7</td>
      <td>False</td>
      <td>all</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>280</td>
      <td>71.264286</td>
      <td>58.5</td>
      <td>47.133473</td>
      <td>0.3</td>
      <td>False</td>
      <td>all</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>298</td>
      <td>71.684564</td>
      <td>56.0</td>
      <td>50.425942</td>
      <td>0.3</td>
      <td>True</td>
      <td>all</td>
    </tr>
    <tr>
      <th>0</th>
      <td>-1</td>
      <td>680</td>
      <td>75.977941</td>
      <td>41.5</td>
      <td>94.338786</td>
      <td>0.7</td>
      <td>True</td>
      <td>circular_neighbors</td>
    </tr>
    <tr>
      <th>0</th>
      <td>-1</td>
      <td>701</td>
      <td>91.649073</td>
      <td>52.0</td>
      <td>101.116296</td>
      <td>0.7</td>
      <td>False</td>
      <td>circular_neighbors</td>
    </tr>
    <tr>
      <th>0</th>
      <td>-1</td>
      <td>711</td>
      <td>99.457103</td>
      <td>44.0</td>
      <td>134.828981</td>
      <td>0.7</td>
      <td>True</td>
      <td>linear_neighbors</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>320</td>
      <td>137.471875</td>
      <td>110.5</td>
      <td>100.269195</td>
      <td>0.3</td>
      <td>True</td>
      <td>circular_neighbors</td>
    </tr>
    <tr>
      <th>0</th>
      <td>-1</td>
      <td>738</td>
      <td>138.127371</td>
      <td>80.0</td>
      <td>153.929957</td>
      <td>0.7</td>
      <td>False</td>
      <td>linear_neighbors</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>299</td>
      <td>148.903010</td>
      <td>118.0</td>
      <td>108.384324</td>
      <td>0.3</td>
      <td>False</td>
      <td>circular_neighbors</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>289</td>
      <td>194.543253</td>
      <td>149.0</td>
      <td>155.471703</td>
      <td>0.3</td>
      <td>True</td>
      <td>linear_neighbors</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>262</td>
      <td>268.610687</td>
      <td>204.5</td>
      <td>199.559191</td>
      <td>0.3</td>
      <td>False</td>
      <td>linear_neighbors</td>
    </tr>
  </tbody>
</table>
</div>




```python
df[df.opinions_prob.isin([0.5])].groupby('opiniao').mean()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>median</th>
      <th>std</th>
      <th>opinions_prob</th>
      <th>random</th>
    </tr>
    <tr>
      <th>opiniao</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>-1</th>
      <td>500.5</td>
      <td>118.515575</td>
      <td>87.666667</td>
      <td>102.448016</td>
      <td>0.5</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>499.5</td>
      <td>115.358659</td>
      <td>85.000000</td>
      <td>98.516492</td>
      <td>0.5</td>
      <td>0.5</td>
    </tr>
  </tbody>
</table>
</div>




```python
df[df.opinions_prob.isin([0.3,0.7])].groupby('opiniao').mean()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>median</th>
      <th>std</th>
      <th>opinions_prob</th>
      <th>random</th>
    </tr>
    <tr>
      <th>opiniao</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>-1</th>
      <td>708.666667</td>
      <td>82.112070</td>
      <td>45.916667</td>
      <td>94.782855</td>
      <td>0.7</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>291.333333</td>
      <td>148.746279</td>
      <td>116.083333</td>
      <td>110.207305</td>
      <td>0.3</td>
      <td>0.5</td>
    </tr>
  </tbody>
</table>
</div>



É possível observar que:

- existe uma relação entre a distribuição de opiniões iniciais e a convergência para uma determinada opinião. Pelos exemplos acima, mantem-se praticamente a mesma proporção entre opiniões iniciais e opinião de consenso após convergência. Quando atores com opiniões -1 possuem a mesma quantidade de atores com opinião 1, 50% das corridas, aproximadamente, tem consenso final em cada opinião. Quando a probabilidade de atribuição de uma opinão muda (caso estudado - 0.7 e 0.3) o numero de corridas que tem consenso na primeira opinião é proximo a 70%.
- Quando existe um desbalanço inicial de opiniões, a média de tempo para convergir para a opinião com maior numero de atores é menor do que o tempo para convergir para a opinião dividida entre menos atores
- Mantendo a proporção de opiniões fixas, o modelo que todos os atores influenciam todos é o que converge mais rápido, seguido pela configuração que todos estão em um circulo e por útlimo a configuração onde estão em fila.
- corridas com opiniões distribuídas randomicamente entre os atores tendem a convergir mais rápido do que quando há segregação de opiniões, para configurações que só existe interações entre vizinhos.



```python

```


```python

```


```python

```


```python

```


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>opiniao</th>
      <th>count</th>
      <th>mean</th>
      <th>median</th>
      <th>std</th>
      <th>opinions_prob</th>
      <th>random</th>
      <th>influence_mode</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-1</td>
      <td>509</td>
      <td>164.836935</td>
      <td>105.0</td>
      <td>173.298440</td>
      <td>0.5</td>
      <td>True</td>
      <td>linear_neighbors</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>491</td>
      <td>144.181263</td>
      <td>101.0</td>
      <td>135.067291</td>
      <td>0.5</td>
      <td>True</td>
      <td>linear_neighbors</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-1</td>
      <td>503</td>
      <td>206.248509</td>
      <td>163.0</td>
      <td>161.835647</td>
      <td>0.5</td>
      <td>False</td>
      <td>linear_neighbors</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>497</td>
      <td>198.372233</td>
      <td>142.0</td>
      <td>168.581619</td>
      <td>0.5</td>
      <td>False</td>
      <td>linear_neighbors</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-1</td>
      <td>711</td>
      <td>99.457103</td>
      <td>44.0</td>
      <td>134.828981</td>
      <td>0.7</td>
      <td>True</td>
      <td>linear_neighbors</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>289</td>
      <td>194.543253</td>
      <td>149.0</td>
      <td>155.471703</td>
      <td>0.3</td>
      <td>True</td>
      <td>linear_neighbors</td>
    </tr>
    <tr>
      <th>6</th>
      <td>-1</td>
      <td>738</td>
      <td>138.127371</td>
      <td>80.0</td>
      <td>153.929957</td>
      <td>0.7</td>
      <td>False</td>
      <td>linear_neighbors</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1</td>
      <td>262</td>
      <td>268.610687</td>
      <td>204.5</td>
      <td>199.559191</td>
      <td>0.3</td>
      <td>False</td>
      <td>linear_neighbors</td>
    </tr>
    <tr>
      <th>8</th>
      <td>-1</td>
      <td>483</td>
      <td>110.942029</td>
      <td>84.0</td>
      <td>92.960086</td>
      <td>0.5</td>
      <td>True</td>
      <td>circular_neighbors</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>517</td>
      <td>106.367505</td>
      <td>80.0</td>
      <td>94.038506</td>
      <td>0.5</td>
      <td>True</td>
      <td>circular_neighbors</td>
    </tr>
    <tr>
      <th>10</th>
      <td>-1</td>
      <td>495</td>
      <td>116.006061</td>
      <td>88.0</td>
      <td>94.079205</td>
      <td>0.5</td>
      <td>False</td>
      <td>circular_neighbors</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1</td>
      <td>505</td>
      <td>122.562376</td>
      <td>98.0</td>
      <td>94.206969</td>
      <td>0.5</td>
      <td>False</td>
      <td>circular_neighbors</td>
    </tr>
    <tr>
      <th>12</th>
      <td>-1</td>
      <td>680</td>
      <td>75.977941</td>
      <td>41.5</td>
      <td>94.338786</td>
      <td>0.7</td>
      <td>True</td>
      <td>circular_neighbors</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1</td>
      <td>320</td>
      <td>137.471875</td>
      <td>110.5</td>
      <td>100.269195</td>
      <td>0.3</td>
      <td>True</td>
      <td>circular_neighbors</td>
    </tr>
    <tr>
      <th>14</th>
      <td>-1</td>
      <td>701</td>
      <td>91.649073</td>
      <td>52.0</td>
      <td>101.116296</td>
      <td>0.7</td>
      <td>False</td>
      <td>circular_neighbors</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1</td>
      <td>299</td>
      <td>148.903010</td>
      <td>118.0</td>
      <td>108.384324</td>
      <td>0.3</td>
      <td>False</td>
      <td>circular_neighbors</td>
    </tr>
    <tr>
      <th>16</th>
      <td>-1</td>
      <td>504</td>
      <td>56.099206</td>
      <td>43.0</td>
      <td>45.719193</td>
      <td>0.5</td>
      <td>True</td>
      <td>all</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1</td>
      <td>496</td>
      <td>62.945565</td>
      <td>45.0</td>
      <td>54.171101</td>
      <td>0.5</td>
      <td>True</td>
      <td>all</td>
    </tr>
    <tr>
      <th>18</th>
      <td>-1</td>
      <td>509</td>
      <td>56.960707</td>
      <td>43.0</td>
      <td>46.795526</td>
      <td>0.5</td>
      <td>False</td>
      <td>all</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1</td>
      <td>491</td>
      <td>57.723014</td>
      <td>44.0</td>
      <td>45.033464</td>
      <td>0.5</td>
      <td>False</td>
      <td>all</td>
    </tr>
    <tr>
      <th>20</th>
      <td>-1</td>
      <td>702</td>
      <td>43.548433</td>
      <td>29.0</td>
      <td>41.695518</td>
      <td>0.7</td>
      <td>True</td>
      <td>all</td>
    </tr>
    <tr>
      <th>21</th>
      <td>1</td>
      <td>298</td>
      <td>71.684564</td>
      <td>56.0</td>
      <td>50.425942</td>
      <td>0.3</td>
      <td>True</td>
      <td>all</td>
    </tr>
    <tr>
      <th>22</th>
      <td>-1</td>
      <td>720</td>
      <td>43.912500</td>
      <td>29.0</td>
      <td>42.787593</td>
      <td>0.7</td>
      <td>False</td>
      <td>all</td>
    </tr>
    <tr>
      <th>23</th>
      <td>1</td>
      <td>280</td>
      <td>71.264286</td>
      <td>58.5</td>
      <td>47.133473</td>
      <td>0.3</td>
      <td>False</td>
      <td>all</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.groupby(['influence_mode', 'random', 'opinions_prob', 'opiniao']).mean()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>median</th>
      <th>std</th>
    </tr>
    <tr>
      <th>influence_mode</th>
      <th>random</th>
      <th>opinions_prob</th>
      <th>opiniao</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="8" valign="top">all</th>
      <th rowspan="4" valign="top">False</th>
      <th>0.3</th>
      <th>1</th>
      <td>280</td>
      <td>71.264286</td>
      <td>58.5</td>
      <td>47.133473</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">0.5</th>
      <th>-1</th>
      <td>509</td>
      <td>56.960707</td>
      <td>43.0</td>
      <td>46.795526</td>
    </tr>
    <tr>
      <th>1</th>
      <td>491</td>
      <td>57.723014</td>
      <td>44.0</td>
      <td>45.033464</td>
    </tr>
    <tr>
      <th>0.7</th>
      <th>-1</th>
      <td>720</td>
      <td>43.912500</td>
      <td>29.0</td>
      <td>42.787593</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">True</th>
      <th>0.3</th>
      <th>1</th>
      <td>298</td>
      <td>71.684564</td>
      <td>56.0</td>
      <td>50.425942</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">0.5</th>
      <th>-1</th>
      <td>504</td>
      <td>56.099206</td>
      <td>43.0</td>
      <td>45.719193</td>
    </tr>
    <tr>
      <th>1</th>
      <td>496</td>
      <td>62.945565</td>
      <td>45.0</td>
      <td>54.171101</td>
    </tr>
    <tr>
      <th>0.7</th>
      <th>-1</th>
      <td>702</td>
      <td>43.548433</td>
      <td>29.0</td>
      <td>41.695518</td>
    </tr>
    <tr>
      <th rowspan="8" valign="top">circular_neighbors</th>
      <th rowspan="4" valign="top">False</th>
      <th>0.3</th>
      <th>1</th>
      <td>299</td>
      <td>148.903010</td>
      <td>118.0</td>
      <td>108.384324</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">0.5</th>
      <th>-1</th>
      <td>495</td>
      <td>116.006061</td>
      <td>88.0</td>
      <td>94.079205</td>
    </tr>
    <tr>
      <th>1</th>
      <td>505</td>
      <td>122.562376</td>
      <td>98.0</td>
      <td>94.206969</td>
    </tr>
    <tr>
      <th>0.7</th>
      <th>-1</th>
      <td>701</td>
      <td>91.649073</td>
      <td>52.0</td>
      <td>101.116296</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">True</th>
      <th>0.3</th>
      <th>1</th>
      <td>320</td>
      <td>137.471875</td>
      <td>110.5</td>
      <td>100.269195</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">0.5</th>
      <th>-1</th>
      <td>483</td>
      <td>110.942029</td>
      <td>84.0</td>
      <td>92.960086</td>
    </tr>
    <tr>
      <th>1</th>
      <td>517</td>
      <td>106.367505</td>
      <td>80.0</td>
      <td>94.038506</td>
    </tr>
    <tr>
      <th>0.7</th>
      <th>-1</th>
      <td>680</td>
      <td>75.977941</td>
      <td>41.5</td>
      <td>94.338786</td>
    </tr>
    <tr>
      <th rowspan="8" valign="top">linear_neighbors</th>
      <th rowspan="4" valign="top">False</th>
      <th>0.3</th>
      <th>1</th>
      <td>262</td>
      <td>268.610687</td>
      <td>204.5</td>
      <td>199.559191</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">0.5</th>
      <th>-1</th>
      <td>503</td>
      <td>206.248509</td>
      <td>163.0</td>
      <td>161.835647</td>
    </tr>
    <tr>
      <th>1</th>
      <td>497</td>
      <td>198.372233</td>
      <td>142.0</td>
      <td>168.581619</td>
    </tr>
    <tr>
      <th>0.7</th>
      <th>-1</th>
      <td>738</td>
      <td>138.127371</td>
      <td>80.0</td>
      <td>153.929957</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">True</th>
      <th>0.3</th>
      <th>1</th>
      <td>289</td>
      <td>194.543253</td>
      <td>149.0</td>
      <td>155.471703</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">0.5</th>
      <th>-1</th>
      <td>509</td>
      <td>164.836935</td>
      <td>105.0</td>
      <td>173.298440</td>
    </tr>
    <tr>
      <th>1</th>
      <td>491</td>
      <td>144.181263</td>
      <td>101.0</td>
      <td>135.067291</td>
    </tr>
    <tr>
      <th>0.7</th>
      <th>-1</th>
      <td>711</td>
      <td>99.457103</td>
      <td>44.0</td>
      <td>134.828981</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
