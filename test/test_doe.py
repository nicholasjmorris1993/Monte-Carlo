import sys
sys.path.append("/home/nick/Monte-Carlo/src")
from doe import xgboost

grid = xgboost(frac=1, repeat=3)

print(grid.combinations)
