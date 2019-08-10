import requests
import pandas as pd


def load_players(_round):
    url = "https://supercoach.heraldsun.com.au/api/bbl/classic/v1/players"
    params = {'round': _round,
              'embed': 'notes,odds,player_stats,positions',
              'xredir': '1'}

    r = requests.get(url,params)
    j = r.json()

    # dest = Path(f'round{_round}.json')
    # if not dest.exists():
    #     with open(dest, 'w') as f:
    #         json.dump(j, f)
            
    df = pd.DataFrame(j)
    cols = ['position', 'other_position', 'cost', 'previous_average']

    df['cost'] = df['player_stats'].str[0].str.get('price')
    df['points'] = df['player_stats'].str[0].str.get('points')
    df['price_change'] = df['player_stats'].str[0].str.get('price_change')
    df['points_thus_far'] = df['player_stats'].str[0].str.get('total_points')
    df['games_played'] = df['player_stats'].str[0].str.get('total_games')
    df['average_points'] = df['points_thus_far'] / df['games_played']
    df['previous_average_rank'] = df['previous_average'].rank(ascending=False)
    
    df['position'] = df['positions'].str[0].str.get('position')
    df['other_position'] = df['positions'].str[1].str.get('position')
    df['team_name'] = df['team'].str.get('abbrev')
    df['name'] = df['first_name'] + ' ' + df['last_name']

    
    df['bwl'] = df['position'].eq('BWL') | df['other_position'].eq('BWL')
    df['bat'] = df['position'].eq('BAT') | df['other_position'].eq('BAT')
    df['wkp'] = df['position'].eq('WKP') | df['other_position'].eq('WKP')
    
    df = df.set_index('name')
    
    return df


def load_games(src='/Users/jackmcivor/bitbucket/cricingesto-scratch/nbs/bbl-fantasy-schedule.csv'):
    schedule = pd.read_csv(src)
    # Start from 1
    schedule.index = range(1, 14)
    return schedule
