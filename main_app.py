from flask import Flask, render_template
from cricket.cricket import cricket_bp
from football.football import football_bp

app = Flask(__name__)

# Register blueprints
app.register_blueprint(cricket_bp, url_prefix='/cricket')
app.register_blueprint(football_bp, url_prefix='/football')

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)