import numpy as np
import pandas as pd


def xgboost(frac, repeat):
    table = XGBoost()
    table.grid()
    table.randomize(frac, repeat)

    return table


class XGBoost:
    def grid(self):
        # set up the parameter options
        n_estimators = [50, 100, 200] 
        learning_rate = [0.001, 0.01, 0.1]
        max_depth = [4, 7, 10]
        min_child_weight = [1, 3, 5]
        colsample_bytree = [0.6, 0.8, 1]
        subsample = [0.6, 0.8, 1]

        # build a grid of all parameter combinations
        self.combinations = np.meshgrid(
            n_estimators,
            learning_rate,
            max_depth,
            min_child_weight,
            colsample_bytree,
            subsample,
        )
        self.combinations = np.reshape(self.combinations, (6, 3**6)).T
        self.combinations = pd.DataFrame(self.combinations, columns=[
            "n_estimators",
            "learning_rate",
            "max_depth",
            "min_child_weight",
            "colsample_bytree",
            "subsample",
        ])

    def randomize(self, frac, repeat):
        # simulate the monte carlo peturbations
        np.random.seed(0)
        simulate = np.random.uniform(
            low=0.5, 
            high=1.5, 
            size=6 * 3**6,
        ).reshape((3**6, 6))

        # add the randomizations to the grid
        combinations_copy = self.combinations.copy()
        for _ in range(repeat - 1):
            combinations_manipulate = combinations_copy.copy()
            self.combinations = pd.concat([
                self.combinations,
                combinations_manipulate * simulate,
            ], axis="index").reset_index(drop=True)

        # ensure parameter boundaries are being respected
        self.combinations["n_estimators"] = np.clip(
            self.combinations["n_estimators"],
            a_min=10,
            a_max=1000,
        )
        self.combinations["learning_rate"] = np.clip(
            self.combinations["learning_rate"],
            a_min=0.0001,
            a_max=1,
        )
        self.combinations["max_depth"] = np.clip(
            self.combinations["max_depth"],
            a_min=2,
            a_max=50,
        )
        self.combinations["min_child_weight"] = np.clip(
            self.combinations["min_child_weight"],
            a_min=1,
            a_max=20,
        )
        self.combinations["colsample_bytree"] = np.clip(
            self.combinations["colsample_bytree"],
            a_min=0.33,
            a_max=1,
        )
        self.combinations["subsample"] = np.clip(
            self.combinations["subsample"],
            a_min=0.33,
            a_max=1,
        )

        # ensure integer parameters have integer values
        self.combinations["n_estimators"] = self.combinations["n_estimators"].round(0).astype("int")
        self.combinations["max_depth"] = self.combinations["max_depth"].round(0).astype("int")
        self.combinations["min_child_weight"] = self.combinations["min_child_weight"].round(0).astype("int")

        # select a random fraction of the grid
        self.combinations = self.combinations.sample(frac=frac, random_state=0).reset_index(drop=True)
