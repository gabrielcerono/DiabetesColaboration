# -*- coding: utf-8 -*-


from sklearn.model_selection import train_test_split
import numpy as np
import sklearn as sk
from sklearn.ensemble import RandomForestRegressor
from random import randint
import pandas as pd

class RankingRE():
  def __init__(self, X, y, loops):
   #X and Y in pandas dataframe.
    self.X = X
    self.y = y
    self.loops = loops

  def ranking_borda(self):
    a = 0
    rankings = np.zeros(len(self.X.columns),)

    for x in range(self.loops):
      seed = randint(0, 10000)
  
  #Splits the train/val set by a seed that generates randomly each loop.
      X_train, X_fr, y_train, y_fr = train_test_split(self.X, self.y, test_size=0.30, random_state= seed)
  #Initializing a random forest
      rf = RandomForestRegressor(n_estimators=100, random_state=30)
  #Fits the Random forest and we calculate a R2. 
      rf.fit(X_train, y_train)
      r2original = rf.score(X_fr, y_fr)
  #We initialize 2 lists to append values from the next loop.
      r2fr= []
      columnsrf= []
  

      for x in self.X.columns:
    
        X_train, X_fr, y_train, y_fr = train_test_split(self.X, self.y, test_size=0.30, random_state = seed)
    #We drop a different column each loop. 
        X_train = X_train.drop([x], axis=1)
        X_fr = X_fr.drop([x], axis=1)
    #We fit our random forest again, but this time our training dataset lacks a feature.
        rf.fit(X_train, y_train)
        r2 = rf.score(X_fr, y_fr)
    #We append to the list each column that we dropped.
        columnsrf.append(x)
    #And we also append, the drop (or gain), in r2 that we got when the feature was missing.
        r2fr.append(r2original - r2)
  
      a += 1 
      outcome = np.array(list(zip(columnsrf, r2fr)))
      outcomepd = pd.DataFrame(data=outcome, columns=['Variables', 'r2-punish'])
      outcomepd['ranking'] = outcomepd['r2-punish'].rank(ascending = False)
      rankings = np.add(outcomepd['ranking'].to_numpy(), rankings)
  
    featuresranks = np.dstack((columnsrf, rankings))
    borda = pd.DataFrame(data = np.squeeze(featuresranks, axis=0), columns=['Categories', 'Borda-Score'])
    borda.sort_values(by='Borda-Score', inplace=True, ascending = True)

    return borda

  def ranking_by_r2_punishment(self):

    
    rankings = np.zeros(len(self.X.columns),)

    for x in range(self.loops):
      seed = randint(0, 10000)
    #Splits the train/val set by a seed that generates randomly each loop.
      X_train, X_fr, y_train, y_fr = train_test_split(self.X, self.y, test_size=0.30, random_state= seed)
    #Initializing a random forest
      rf = RandomForestRegressor(n_estimators=100, random_state=30)
  #Fits the Random forest and we calculate a R2. 
      rf.fit(X_train, y_train)
      r2original = rf.score(X_fr, y_fr)
  #We initialize 2 lists to append values from the next loop.
      r2fr= []
      columnsrf= []
  

      for x in self.X.columns:

        X_train, X_fr, y_train, y_fr = train_test_split(self.X, self.y, test_size=0.30, random_state = seed)
    #We drop a different column each loop.
        X_train = X_train.drop([x], axis=1)
        X_fr = X_fr.drop([x], axis=1)
    #We fit our random forest again, but this time our training dataset lacks a feature.
        rf.fit(X_train, y_train)
        r2 = rf.score(X_fr, y_fr)
    #We append to the list each column that we dropped.
        columnsrf.append(x)
    #And we also append, the drop (or gain), in r2 that we got when the feature was missing.
        r2fr.append(r2original - r2)

      outcome = np.array(r2fr)
      rankings = np.add(outcome, rankings)
    
    rankings = np.true_divide(rankings, self.loops)
    featuresranks = np.dstack((columnsrf, rankings))
    borda = pd.DataFrame(data = np.squeeze(featuresranks, axis=0), columns=['Categories', 'r2-punish'])
    borda['ranking'] = borda['r2-punish'].rank(ascending = False)

    return borda
