# -*- coding: utf-8 -*-
"""gabrielrfev5.ipynb


"""

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

  def rankingborda(self):
    a = 0
    rankings = np.zeros(len(self.X.columns),)

    for x in range(self.loops):
      seed = randint(0, 10000)
  #Aca tendria que venir el x in range(100):
  #Spliteas el test 
      X_train, X_fr, y_train, y_fr = train_test_split(self.X, self.y, test_size=0.30, random_state= seed)
  #Inicializas un random forest
      rf = RandomForestRegressor(n_estimators=100, random_state=30)
  # Lo fiteas, y calculas el r2
      rf.fit(X_train, y_train)
      r2original = rf.score(X_fr, y_fr)
  #Le inicializas esto pa hacer columnas
      r2fr= []
      columnsrf= []
  

      for x in self.X.columns:
    #Si no le pones el train, perdes columnas y no las regeneras
        X_train, X_fr, y_train, y_fr = train_test_split(self.X, self.y, test_size=0.30, random_state = seed)
    #Dropeas, columna por columna. 
        X_train = X_train.drop([x], axis=1)
        X_fr = X_fr.drop([x], axis=1)
    #Fiteas y sacas el score
        rf.fit(X_train, y_train)
        r2 = rf.score(X_fr, y_fr)
    #Apendeas cada columnas sacada
        columnsrf.append(x)
    #Apendeas la diferencia r2
        r2fr.append(r2original - r2)
  #La idea seria hacer en cada loop, el dataset ese que hice, pero la columna con los rangos indexarla a otro dataframe. 
      a += 1 
      resultado = np.array(list(zip(columnsrf, r2fr)))
      resultadopd = pd.DataFrame(data=resultado, columns=['Variables', 'r2-punish'])
      resultadopd['ranking'] = resultadopd['r2-punish'].rank(ascending = False)
      rankings = np.add(resultadopd['ranking'].to_numpy(), rankings)
  
    featuresranks = np.dstack((columnsrf, rankings))
    borda = pd.DataFrame(data = np.squeeze(featuresranks, axis=0), columns=['Categories', 'Borda-Score'])

    return borda

  def rankingr2(self):

    
    rankings = np.zeros(len(self.X.columns),)

    for x in range(self.loops):
      seed = randint(0, 10000)
  #Aca tendria que venir el x in range(100):
  #Spliteas el test 
      X_train, X_fr, y_train, y_fr = train_test_split(self.X, self.y, test_size=0.30, random_state= seed)
  #Inicializas un random forest
      rf = RandomForestRegressor(n_estimators=100, random_state=30)
  # Lo fiteas, y calculas el r2
      rf.fit(X_train, y_train)
      r2original = rf.score(X_fr, y_fr)
  #Le inicializas esto pa hacer columnas
      r2fr= []
      columnsrf= []
  

      for x in self.X.columns:
    #Si no le pones el train, perdes columnas y no las regeneras
        X_train, X_fr, y_train, y_fr = train_test_split(self.X, self.y, test_size=0.30, random_state = seed)
    #Dropeas, columna por columna. 
        X_train = X_train.drop([x], axis=1)
        X_fr = X_fr.drop([x], axis=1)
    #Fiteas y sacas el score
        rf.fit(X_train, y_train)
        r2 = rf.score(X_fr, y_fr)
    #Apendeas cada columnas sacada
        columnsrf.append(x)
    #Apendeas la diferencia r2
        r2fr.append(r2original - r2)
  #La idea seria hacer en cada loop, el dataset ese que hice, pero la columna con los rangos indexarla a otro dataframe.  
      resultado = np.array(r2fr)
      rankings = np.add(resultado, rankings)
    
    rankings = np.true_divide(rankings, self.loops)
    featuresranks = np.dstack((columnsrf, rankings))
    borda = pd.DataFrame(data = np.squeeze(featuresranks, axis=0), columns=['Categories', 'r2-punish'])
    borda['ranking'] = borda['r2-punish'].rank(ascending = False)

    return borda
