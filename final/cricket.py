from flask import Blueprint, render_template, request
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import joblib
import os
from typing import List, Tuple

# Create Blueprint instead of Flask app
cricket_bp = Blueprint('cricket', __name__, 
                      template_folder='templates',
                      static_folder='static',
                      static_url_path='/cricket/static')

class FantasyTeamSelector:
    def __init__(self, data_merge_path: str, venue_agg_path: str, recent_agg_path: str, 
                 player_data_path: str, matchup_data_path: str):
        """
        Initialize the selector with paths to all required data files.
        """
        # Load all datasets
        self.player_data = pd.read_csv(player_data_path)
        self.matchup_data = pd.read_csv(matchup_data_path)
        self.venue_agg = pd.read_csv(venue_agg_path)
        self.recent_agg = pd.read_csv(recent_agg_path)
        self.data_merge = pd.read_csv(data_merge_path)
        
        # Initialize models and scalers
        self.batsman_scaler = StandardScaler()
        self.bowler_scaler = StandardScaler()
        self.allrounder_scaler = StandardScaler()
        
        self.batsman_perf_model = None
        self.batsman_matchup_model = None
        self.bowler_perf_model = None
        self.bowler_matchup_model = None
        self.allrounder_perf_model = None
        self.allrounder_matchup_model = None
        
        # Preprocess data and train models
        self._preprocess_data()
        self._train_models()
    
    def _preprocess_data(self):
        """Preprocess data for all player types."""
        # Batsmen data
        self.batsmen_data = self.data_merge.merge(self.venue_agg, on=['Player Name1', 'Venue'], how='left')\
                                          .merge(self.recent_agg, on='Player Name1', how='left')
        batsmen = self.player_data[self.player_data["Role"].str.contains('Batter|Wicketkeeper', na=False)]
        self.batsmen_data = self.batsmen_data[self.batsmen_data['Player Name1'].isin(batsmen['Player Name'])]
        self.batsmen_data = self.batsmen_data.rename(
            columns={'avg_batting_points_x': 'avg_batting_points', 'recent_avg_batting_x': 'recent_avg_batting'}
        ).drop(columns=['avg_batting_points_y', 'recent_avg_batting_y'], errors='ignore')
        self.batsmen_features = ['Runs Scored', 'Fantasy Points', 'Fours', 'Sixes', 
                               'avg_batting_points', 'recent_avg_batting', 
                               'Avg Runs per Over', 'Boundary %']
        self.X_batsmen = self.batsmen_data[self.batsmen_features].fillna(0)
        self.X_batsmen_scaled = self.batsman_scaler.fit_transform(self.X_batsmen)
        
        # Bowlers data
        self.bowlers_data = self.data_merge.merge(self.venue_agg, on=['Player Name1', 'Venue'], how='left')\
                                          .merge(self.recent_agg, on='Player Name1', how='left')
        self.bowlers_features = ['Wickets', 'Fantasy Points', 'Economy', 
                               'avg_bowling_points_x', 'recent_avg_bowling_x']
        self.X_bowlers = self.bowlers_data[self.bowlers_features].fillna(0)
        self.X_bowlers_scaled = self.bowler_scaler.fit_transform(self.X_bowlers)
        
        # All-rounders data
        self.allrounders_data = self.data_merge.merge(self.venue_agg, on=['Player Name1', 'Venue'], how='left')\
                                              .merge(self.recent_agg, on='Player Name1', how='left')
        self.allrounders_list = self.player_data[self.player_data["Role"].str.contains('Allrounder', na=False)]['Player Name'].tolist()
        self.allrounders_features = ['Fantasy Points', 'avg_batting_points_x', 'recent_avg_batting_x', 
                                   'avg_bowling_points_x', 'recent_avg_bowling_x']
        self.X_allrounders = self.allrounders_data[self.allrounders_features].fillna(0)
        self.X_allrounders_scaled = self.allrounder_scaler.fit_transform(self.X_allrounders)
    
    def _train_models(self):
        """Train all required models."""
        # Batsmen models
        y_batsmen = self.batsmen_data['Fantasy Points']
        X_train, _, y_train, _ = train_test_split(self.X_batsmen_scaled, y_batsmen, test_size=0.2, random_state=42)
        self.batsman_perf_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
        self.batsman_perf_model.fit(X_train, y_train)
        
        # Bowlers models
        y_bowlers = self.bowlers_data['Fantasy Points']
        X_train, _, y_train, _ = train_test_split(self.X_bowlers_scaled, y_bowlers, test_size=0.2, random_state=42)
        self.bowler_perf_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
        self.bowler_perf_model.fit(X_train, y_train)
        
        # All-rounders models
        y_allrounders = self.allrounders_data['Fantasy Points']
        X_train, _, y_train, _ = train_test_split(self.X_allrounders_scaled, y_allrounders, test_size=0.2, random_state=42)
        self.allrounder_perf_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
        self.allrounder_perf_model.fit(X_train, y_train)
    
    def select_best_team(self, team1: List[str], team2: List[str], venue: str, 
                        batsmen_count: int = 5, bowlers_count: int = 5, allrounders_count: int = 1):
        """
        Select the best team based on input parameters.
        """
        # Get best players from each category
        best_batsmen = self._select_best_batsmen(team1, team2, venue)[:batsmen_count]
        best_bowlers = self._select_best_bowlers(team1, team2, venue)[:bowlers_count]
        best_allrounders = self._select_best_allrounders(team1, team2, venue)[:allrounders_count]
        
        # Combine all players
        combined_players = best_batsmen + best_bowlers + best_allrounders
        combined_players.sort(key=lambda x: x[1], reverse=True)
        
        return {
            'batsmen': [player[0] for player in best_batsmen],
            'bowlers': [player[0] for player in best_bowlers],
            'allrounders': [player[0] for player in best_allrounders],
            'combined_team': [player[0] for player in combined_players]
        }
    def _select_best_batsmen(self, team1: List[str], team2: List[str], venue: str) -> List[Tuple[str, float]]:
        """Select best batsmen."""
        all_batsmen = team1 + team2
        batsman_scores = {}
        
        for batsman in all_batsmen:
            player_data = self.batsmen_data[self.batsmen_data['Player Name1'] == batsman]
            if player_data.empty:
                continue
            
            player_features = player_data[self.batsmen_features].fillna(0)
            player_features_scaled = self.batsman_scaler.transform(player_features)
            performance_score = self.batsman_perf_model.predict(player_features_scaled)[0]
            
            matchup_score = self._get_batsman_matchup_score(batsman, all_batsmen)
            
            recent_form_weight = 0.3
            venue_weight = 0.4
            matchup_weight = 0.3
            
            recent_form_score = player_data['recent_avg_batting'].mean() * recent_form_weight
            venue_score = player_data[player_data['Venue'] == venue]['avg_batting_points'].mean() * venue_weight
            
            total_score = performance_score + recent_form_score + venue_score + (matchup_score * matchup_weight)
            batsman_scores[batsman] = total_score
        
        return sorted(batsman_scores.items(), key=lambda x: x[1], reverse=True)
    
    def _select_best_bowlers(self, team1: List[str], team2: List[str], venue: str) -> List[Tuple[str, float]]:
        """Select best bowlers."""
        team1_batsmen = self.player_data[(self.player_data['Player Name'].isin(team1)) & 
                                       ~self.player_data['Role'].str.contains('Bowler', na=False)]['Player Name'].tolist()
        team2_batsmen = self.player_data[(self.player_data['Player Name'].isin(team2)) & 
                                       ~self.player_data['Role'].str.contains('Bowler', na=False)]['Player Name'].tolist()
        
        bowlers = self.player_data[self.player_data['Role'].str.contains('Bowler', na=False)]['Player Name'].tolist()
        all_bowlers = [player for player in team1 + team2 if player in bowlers]
        
        bowler_scores = {}
        for bowler in all_bowlers:
            player_data = self.bowlers_data[self.bowlers_data['Player Name1'] == bowler]
            if player_data.empty:
                continue
            
            player_features = player_data[self.bowlers_features].fillna(0)
            player_features_scaled = self.bowler_scaler.transform(player_features)
            performance_score = self.bowler_perf_model.predict(player_features_scaled)[0]
            
            opposition_batsmen = team2_batsmen if bowler in team1 else team1_batsmen
            matchup_score = self._get_bowler_matchup_score(bowler, opposition_batsmen) * 0.3
            
            venue_score = player_data[player_data['Venue'] == venue]['avg_bowling_points_x'].mean() * 0.4
            recent_form_score = player_data['recent_avg_bowling_x'].mean() * 0.3
            
            total_score = performance_score + matchup_score + venue_score + recent_form_score
            bowler_scores[bowler] = total_score
        
        return sorted(bowler_scores.items(), key=lambda x: x[1], reverse=True)
    
    def _select_best_allrounders(self, team1: List[str], team2: List[str], venue: str) -> List[Tuple[str, float]]:
        """Select best all-rounders."""
        allrounders = [player for player in team1 + team2 if player in self.allrounders_list]
        opposition_players = [player for player in team1 + team2 if player not in allrounders]
        
        allrounder_scores = {}
        for player in allrounders:
            player_data = self.allrounders_data[self.allrounders_data['Player Name1'] == player]
            if player_data.empty:
                continue
            
            player_features = player_data[self.allrounders_features].fillna(0)
            player_features_scaled = self.allrounder_scaler.transform(player_features)
            performance_score = self.allrounder_perf_model.predict(player_features_scaled)[0]
            
            matchup_score = self._get_allrounder_matchup_score(player, opposition_players) * 0.3
            
            venue_score = player_data[player_data['Venue'] == venue]['avg_batting_points_x'].mean() * 0.2
            recent_form_score = player_data['recent_avg_batting_x'].mean() * 0.2
            
            total_score = performance_score + matchup_score + venue_score + recent_form_score
            allrounder_scores[player] = total_score
        
        return sorted(allrounder_scores.items(), key=lambda x: x[1], reverse=True)
    
    def _get_batsman_matchup_score(self, batsman: str, opposition_bowlers: List[str]) -> float:
        """Get matchup score for a batsman."""
        matchup_scores = self.matchup_data[(self.matchup_data["Batsman"] == batsman) &
                                         (self.matchup_data["Bowler"].isin(opposition_bowlers))]["Batsman Points"]
        return matchup_scores.mean() if not matchup_scores.empty else 0
    
    def _get_bowler_matchup_score(self, bowler: str, opposition_batsmen: List[str]) -> float:
        """Get matchup score for a bowler."""
        matchup_scores = self.matchup_data[(self.matchup_data["Bowler"] == bowler) &
                                         (self.matchup_data["Batsman"].isin(opposition_batsmen))]["Bowler Points"]
        return matchup_scores.mean() if not matchup_scores.empty else 0
    
    def _get_allrounder_matchup_score(self, allrounder: str, opposition_players: List[str]) -> float:
        """Get matchup score for an all-rounder."""
        matchup_scores = self.matchup_data[(self.matchup_data['Batsman'] == allrounder) &
                                         (self.matchup_data['Bowler'].isin(opposition_players))]
        return matchup_scores['Batsman Points'].mean() if not matchup_scores.empty else 0

    
    # [Keep all the remaining methods unchanged from your original code]

# Initialize the selector when the blueprint is registered
selector = None

@cricket_bp.record_once
def on_load(state):
    global selector
    base_dir = os.path.dirname(state.app.root_path)  # Get project root directory
    cricket_data_dir = os.path.join(base_dir, 'cric_fin/cricket/data')

    data_paths = {
        'data_merge_path': os.path.join(cricket_data_dir, 'data_merge.csv'),
        'venue_agg_path': os.path.join(cricket_data_dir, 'venue_aggregated.csv'),
        'recent_agg_path': os.path.join(cricket_data_dir, 'recent_aggregated.csv'),
        'player_data_path': os.path.join(cricket_data_dir, 'player_data_cleaned.csv'),
        'matchup_data_path': os.path.join(cricket_data_dir, 'cleaned_matchup.csv')
    }

    selector = FantasyTeamSelector(**data_paths)

@cricket_bp.route('/', methods=['GET', 'POST'])
def cricket_home():
    if request.method == 'POST':
        # Get form data
        team1 = request.form.get('team1', '').split(',')
        team2 = request.form.get('team2', '').split(',')
        venue = request.form.get('venue', '')
        batsmen_count = int(request.form.get('batsmen_count', 5))
        bowlers_count = int(request.form.get('bowlers_count', 5))
        allrounders_count = int(request.form.get('allrounders_count', 1))
        
        # Clean inputs
        team1 = [name.strip() for name in team1 if name.strip()]
        team2 = [name.strip() for name in team2 if name.strip()]
        
        # Get best team
        results = selector.select_best_team(
            team1, team2, venue,
            batsmen_count=batsmen_count,
            bowlers_count=bowlers_count,
            allrounders_count=allrounders_count
        )
        
        return render_template('cricket.html', results=results, 
                             batsmen_count=batsmen_count,
                             bowlers_count=bowlers_count,
                             allrounders_count=allrounders_count,
                             team1=', '.join(team1),
                             team2=', '.join(team2),
                             venue=venue)
    
    return render_template('cricket.html')