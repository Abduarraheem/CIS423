import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from sklearn.base import BaseEstimator, TransformerMixin #gives us the tools to build custom transformers

#This class maps values in a column, numeric or categorical.
class MappingTransformer(BaseEstimator, TransformerMixin):
  
  def __init__(self, mapping_column, mapping_dict:dict):  
    self.mapping_dict = mapping_dict
    self.mapping_column = mapping_column  #column to focus on

  def fit(self, X, y = None):
    print("Warning: MappingTransformer.fit does nothing.")
    return X

  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'MappingTransformer.transform expected Dataframe but got {type(X)} instead.'
    assert self.mapping_column in X.columns.to_list(), f'MappingTransformer.transform unknown column {self.mapping_column}'
    X_ = X.copy()
    X_[self.mapping_column].replace(self.mapping_dict, inplace=True)
    return X_

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result
 
#This class will rename one or more columns.
class RenamingTransformer(BaseEstimator, TransformerMixin):
  #your __init__ method below
    def __init__(self, renamed_dict:dict):  
      self.renamed_dict = renamed_dict
  #write the transform method without asserts. Again, maybe copy and paste from MappingTransformer and fix up.
    def transform(self, X):
      assert isinstance(X, pd.core.frame.DataFrame), f'RenamingTransformer.transform expected Dataframe but got {type(X)} instead.'
      #your assert code below
      assert set(self.renamed_dict.keys()).issubset(X.columns), f"{set(self.renamed_dict.keys())-set(X.columns)} not a column."
      X_ = X.copy()
      X_.rename(columns=self.renamed_dict) 
      return X_

class OHETransformer(BaseEstimator, TransformerMixin):
  def __init__(self, target_column, dummy_na=False, drop_first=True):  
    self.target_column = target_column
    self.dummy_na = dummy_na
    self.drop_first = drop_first
  
  #fill in the rest below
  def fit(self, X, y = None):
    print("Warning: OHETransformer.fit does nothing.")
    return X
  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'OHETransformer.transform expected Dataframe but got {type(X)} instead.'
    assert self.target_column in X.columns.to_list(), f'OHETransformer.transform unknown column {self.target_column}'
    X_ = X.copy()
    X_ = pd.get_dummies(X_, prefix=self.target_column, prefix_sep='_', 
                    columns=[self.target_column],
                    dummy_na=self.dummy_na,       #will try to impute later so leave NaNs in place
                    drop_first=self.drop_first    #really should be True but I wanted to give a clearer picture
                  )
    return X_

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result
  
class DropColumnsTransformer(BaseEstimator, TransformerMixin):
  def __init__(self, column_list, action='drop'):
    assert action in ['keep', 'drop'], f'DropColumnsTransformer action {action} not in ["keep", "drop"]'
    assert isinstance(column_list, list), f'DropColumnsTransformer expected list but saw {type(column_list)}'
    self.column_list = column_list
    self.action = action

  #fill in rest below
  def fit(self, X, y = None):
    print("Warning: DropColumnsTransformer.fit does nothing.")
    return X

  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'DropColumnsTransformer.transform expected Dataframe but got {type(X)} instead.'
    X_ = X.copy()
    columns = [col for col in self.column_list if col in X_.columns]
    assert set(self.column_list).issubset(columns), f"{set(self.column_list)-set(columns)} not in columns."
    if self.action == 'drop':
      X_ = X_.drop(columns=columns)
    else:
      X_ = X_[columns]
    return X_

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result

class Sigma3Transformer(BaseEstimator, TransformerMixin):
  def __init__(self, target_column):  
    self.target_column = target_column
  
  def fit(self, X, y = None):
    print("Warning: Sigma3Transformer.fit does nothing.")
    return X

  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'Sigma3Transformer.transform expected Dataframe but got {type(X)} instead.'
    X_ = X.copy()
    assert self.target_column in X_.columns.to_list(), f'unknown column {self.target_column}'
    assert all([isinstance(v, (int, float)) for v in X_[self.target_column].to_list()])

    #compute mean of column - look for method
    m = X_[self.target_column].mean()
    #compute std of column - look for method
    sigma = X_[self.target_column].std()
    X_[self.target_column] = X_[self.target_column].clip(lower =m-3*sigma, upper=m+3*sigma)
    return X_

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result

class TukeyTransformer(BaseEstimator, TransformerMixin):
  def __init__(self, target_column, fence='outer'):
    assert fence in ['inner', 'outer']
    self.target_column = target_column
    self.fence = fence
    
  def fit(self, X, y = None):
    print("Warning: TukeyTransformer.fit does nothing.")
    return X

  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'TukeyTransformer.transform expected Dataframe but got {type(X)} instead.'
    X_ = X.copy()
    assert self.target_column in X_.columns.to_list(), f'unknown column {self.target_column}'
    assert all([isinstance(v, (int, float)) for v in X_[self.target_column].to_list()])

    #now add on outer fences
    q1 = X_[self.target_column].quantile(0.25)
    q3 = X_[self.target_column].quantile(0.75)
    iqr = q3-q1
    inner_low = q1-1.5*iqr
    inner_high = q1+1.5*iqr
    outer_low = q1-3*iqr
    outer_high = q3+3*iqr
    if self.fence == "outer":
      X_[self.target_column] = X_[self.target_column].clip(lower=outer_low, upper=outer_high)
    else:
      X_[self.target_column] = X_[self.target_column].clip(lower=inner_low, upper=inner_high)

    return X_
  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result
