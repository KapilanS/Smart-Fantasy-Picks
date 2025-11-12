from flask import Blueprint, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from sklearn.preprocessing import LabelEncoder
import re
import os

# Create Blueprint instead of Flask app
football_bp = Blueprint('football', __name__,
                      template_folder='templates',
                      static_folder='static',
                      static_url_path='/football/static')

# Initialize model when blueprint is registered
model = None

@football_bp.record_once
def on_load(state):
    global model
    try:
        # Get the path to the model file
        base_dir = os.path.dirname(state.app.root_path)
        model_path = os.path.join(base_dir, 'CRIC_FIN/football/model/fantasy_points_model.h5')
        
        model = load_model(model_path, 
                         custom_objects={'mse': MeanSquaredError})
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        model = None

def parse_player_input(input_text):
    players = []
    pattern = r'\(\s*([^,]+?)\s*,\s*([^,]+?)\s*,\s*([^,]+?)\s*\)'
    
    matches = re.findall(pattern, input_text)
    for match in matches:
        name, position, home_away = match
        position = position.upper()
        home_away = home_away.lower()
        
        if position not in ['GK', 'DEF', 'MID', 'FWD']:
            continue
        if home_away not in ['home', 'away']:
            continue
            
        players.append((name.strip(), position, 1 if home_away == 'home' else 0))
    
    return players

def select_fantasy_team(players, model):
    if len(players) < 11:
        return []

    player_names = [p[0] for p in players]
    positions = [p[1] for p in players]
    home_status = [p[2] for p in players]
    
    encoder = LabelEncoder()
    player_ids = encoder.fit_transform(player_names)
    
    points = model.predict([np.array(player_ids), np.array(home_status)]).flatten()
    
    player_data = list(zip(player_names, positions, home_status, points))
    player_data.sort(key=lambda x: x[3], reverse=True)
    
    team = []
    slots = {'GK': 1, 'DEF': 5, 'MID': 5, 'FWD': 3}

    for player in player_data:
        name, pos, home, pts = player
        if slots[pos] > 0:
            team.append((name, pos, 'Home' if home else 'Away', pts))
            slots[pos] -= 1
            if sum(slots.values()) == 0:
                break

    return team[:11]

def format_team_output(team):
    if not team:
        return {}

    captain = team[0]
    vice_captain = team[1]
    
    formatted_team = []
    total_points = 0

    for i, (name, pos, ha, pts) in enumerate(team):
        if i == 0:  
            pts *= 2
            role = "[C]"
        elif i == 1:  
            pts *= 1.5
            role = "[VC]"
        else:
            role = ""

        formatted_team.append({
            'name': name,
            'position': pos,
            'home_away': ha,
            'points': f"{pts:.1f}",
            'role': role
        })
        total_points += pts

    return {
        "players": formatted_team,
        "total_points": f"{total_points:.1f}",
        "captain": captain[0],
        "vice_captain": vice_captain[0]
    }

@football_bp.route("/", methods=["GET", "POST"])
def football_home():
    results = None
    input_value = ""

    if request.method == "POST":
        input_value = request.form.get("players", "")
        players = parse_player_input(input_value)

        if not model:
            results = {"error": "Model failed to load. Please try again later."}
        elif len(players) < 11:
            results = {"error": "You need at least 11 players to form a team."}
        else:
            team = select_fantasy_team(players, model)
            results = format_team_output(team)

    return render_template("football.html", 
                         results=results,
                         input_value=input_value)