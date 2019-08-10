"""
# Options
# Don't have functions and allow everything to be in the same scope
# Pass all variables as arguments
# Let all variables be attributes on the object
# Do some funky unpacking like:
# prob = self.prob
# xs, xt, xc, t = self.vars
# players, positions, rounds = self.indices
# costs, playing_positions, values = self.data
"""
from collections import namedtuple, defaultdict

import pulp
from pulp import LpProblem, LpMaximize, LpVariable, lpSum
import pandas as pd

from fantasy import io

# variables = ['xs', 'xt', 'xc' 't']
Variables = namedtuple('Variables', ['xs', 'xt', 'xc', 't'])
Data = namedtuple('Data', ['costs', 'values', 'playing_positions'])
Indices = namedtuple('Indices', ['players', 'positions', 'rounds'])
# Parameters = 


class BBLFantasy:
    def __init__(self, budget=2_000_000, n_trading_rounds=None, first_round=1, last_round=14, max_trades_allowed=3):
        self.prob = LpProblem('BBLFantasy', LpMaximize)
        
        # Parameters
        self.n_trading_rounds = n_trading_rounds
        self.budget = budget
        self.first_round = first_round
        self.last_round = last_round
        self.max_trades_allowed = max_trades_allowed

    def collect_data(self):
        # read in games, availability
        # get df from the network
        df = io.load_players(_round=self.last_round)
        df['expected_points'] = df['average_points']
        df = df.dropna(subset=['expected_points', 'team_name', 'cost'])
        games = io.load_games()

        return df, games


    def define_data(self, df, games, availability=None):
        """
        df has index players, columns ['expected_points', 'cost', 'bat', 'bwl', 'wkp']
        """
        # Indices
        players = tuple(df.index)
        positions = ('bwl', 'bat', 'wkp')
        rounds = tuple(range(self.first_round, self.last_round + 1))  # 13 is the last round
        nrounds = len(rounds)

        if self.last_round > 13:
            raise ValueError(f'The last round ({self.last_round}) cannot be above 13!')

        if self.n_trading_rounds is None:
            self.n_trading_rounds = nrounds - 1

        if self.n_trading_rounds > nrounds - 1:
            raise ValueError(f'There are {nrounds} rounds and {self.n_trading_rounds} trading rounds!')

        # Variables
        values = defaultdict(dict)  # players, rounds
        costs = {}  # players
        playing_positions = defaultdict(dict)  # players, positions

        for i in players:
            team = df.loc[i, 'team_name']
            for r in rounds:
                values[i][r] = df.loc[i, 'expected_points'] * games.loc[r, team]

        for i in players:
            for p in positions:
                playing_positions[i][p] = df.loc[i, p]

        for i in players:
            costs[i] = df.loc[i, 'cost']

        return (players, positions, rounds,
                costs, values, playing_positions)

    def define_vars(self, players, positions, rounds):
        # Decision variables
        # Squad
        xs = LpVariable.dicts('xs', (players, positions, rounds), 0, 1, cat='Integer')
        # Team
        xt = LpVariable.dicts('xt', (players, positions, rounds), 0, 1, cat='Integer')
        # Captain (is positionless to remove a few decision variables)
        xc = LpVariable.dicts('xc', (players, rounds), 0, 1, cat='Integer')

        # Trades- ignore the last round
        t = LpVariable.dicts('t', (players, rounds[:self.n_trading_rounds]), 0, 1, cat='Integer')

        return xs, xt, xc, t
        # return Variables(xs, xt, xc, t)

    def define_obj(self, xt, xc, values, players, positions, rounds):
        self.prob += (lpSum(values[i][r]*xt[i][p][r] for i in players for p in positions for r in rounds)
                      + lpSum(values[i][r]*xc[i][r] for i in players for r in rounds)) / len(rounds)

    def define_constraints(self, xs, xt, xc, t, players, positions, rounds, costs, playing_positions):
        for r in rounds:
            # Limit budget each round
            self.prob += lpSum(costs[i]*xs[i][p][r] for i in players for p in positions) <= self.budget

            # The squad must have 7 batters, 7 bowlers and 2 wkp (16 players)
            self.prob += lpSum(xs[i]['bwl'][r] for i in players) == 7
            self.prob += lpSum(xs[i]['bat'][r] for i in players) == 7
            self.prob += lpSum(xs[i]['wkp'][r] for i in players) == 2
            
            # The team must have 5 batters, 5 bowlers, 1 wkp (11 players)
            self.prob += lpSum(xt[i]['bwl'][r] for i in players) == 5
            self.prob += lpSum(xt[i]['bat'][r] for i in players) == 5
            self.prob += lpSum(xt[i]['wkp'][r] for i in players) == 1
            
            # One captain
            self.prob += lpSum(xc[i][r] for i in players) == 1
        
        for r in rounds:
            for i in players:
                # Each player can only be picked in one position
                self.prob += lpSum(xs[i][p][r] for p in positions) <= 1
                
                # Captain must be in the team
                # self.prob += xt[i][p][r] >= xc[i][p][r]  # This constraint only works if captains have positions, but seems to be faster
                self.prob += lpSum(xt[i][p][r] for p in positions) >= xc[i][r]


        for r in rounds:
            for i in players:
                for p in positions:
                    # Players must actually play in the position they are picked
                    self.prob += xs[i][p][r] <= playing_positions[i][p]
                    self.prob += xt[i][p][r] <= playing_positions[i][p]
                    
                    # Players in the team must be in the squad
                    self.prob += xs[i][p][r] >= xt[i][p][r]
                    
                    # If their value is 0, don't pick them! (heuristic to speed up solving)
                    # self.prob += xt[i][p][r] >= values[i][r]
                    
        # Trading
        for r in rounds[:self.n_trading_rounds]:  # rounds[:-1]:
            for i in players:
                self.prob += lpSum(xs[i][p][r] for p in positions) + t[i][r] >= lpSum(xs[i][p][r+1] for p in positions)
                self.prob += lpSum(xs[i][p][r+1] for p in positions) + t[i][r] >= lpSum(xs[i][p][r] for p in positions)
                
                # These constraints disallow wasted trades, which just helps with speed
                self.prob += lpSum(xs[i][p][r+1] for p in positions) + t[i][r] + lpSum(xs[i][p][r] for p in positions) <= 2
                self.prob += t[i][r] <= lpSum(xs[i][p][r] + xs[i][p][r+1] for p in positions)
                
                    
        for r in rounds[:self.n_trading_rounds]:  # rounds[:-1]:
            # The trading variable is used for both trades in and trades out, so we need to limit by twice the maximum number of trades
            # For speed, it may be best to assume we will always make 3 trades (equality constraint)
            self.prob += lpSum(t[i][r] for i in players) <= 2*self.max_trades_allowed
            

        # The following constraint is only needed if there are non-trading rounds 
        # If trades are not allowed, then the squad must stay the same (but players may change position or be benched)
        for r in rounds[self.n_trading_rounds:-1]:
            for i in players:
                self.prob += lpSum(xs[i][p][r] for p in positions) == lpSum(xs[i][p][r+1] for p in positions)

    # def compile(self):
    #     df, games = self.collect_data()

    #     players, positions, rounds, costs, values, playing_positions = self.define_data(df, games)
    #     xs, xt, xc, t = self.define_vars(players, positions, rounds)
    #     self.define_obj(xt, xc, values, players, positions, rounds)
    #     self.define_constraints(xs, xt, xc, t, players, positions, rounds, costs, playing_positions)

    # def solve(self):
    #     """Solves the problem and spits out the solution
    #     """
    #     self.prob.solve()
    #     return parse_solution()

    def compile_and_solve(self):
        df, games = self.collect_data()

        players, positions, rounds, costs, values, playing_positions = self.define_data(df, games)
        xs, xt, xc, t = self.define_vars(players, positions, rounds)
        self.define_obj(xt, xc, values, players, positions, rounds)
        self.define_constraints(xs, xt, xc, t, players, positions, rounds, costs, playing_positions)
        # import ipdb; ipdb.set_trace()
        self.prob.solve()
        return parse_solution(df, players, positions, rounds, xs, xt, xc, t)


def parse_solution(df, players, positions, rounds, xs, xt, xc, t):
    gen = itervars(df, players, positions, rounds, xs, xt, xc, t)
    return pd.DataFrame(gen).set_index(['team', 'name', 'round'])['pos'].unstack('round').fillna('').sort_index()


def itervars(df, players, positions, rounds, xs, xt, xc, t):
    for i in players:
        for p in positions:
            for r in rounds:
                in_squad = pulp.value(xs[i][p][r])
                if not in_squad:
                    continue

                in_team = pulp.value(xt[i][p][r])
                is_captain = pulp.value(xc[i][r])
                # is_traded = pl.value(t[i][r])

                team = df.loc[i, 'team_name']                

                if is_captain:
                    pos = f'{p}©'
                elif in_squad and not in_team:
                    pos = f'sub({p})'
                elif in_team:
                    pos = p

                yield {'name': i, 'team': team, 'round': r, 'pos': pos}
