from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/authority')
def authority():
    # Mock data for authority dashboard
    stats = {
        "active_cases": 1245,
        "risk_level": "High",
        "hospital_capacity": "85%"
    }
    alerts = [
        {"id": 1, "level": "Critical", "message": "High viral load detected in Downtown district.", "time": "10 mins ago"},
        {"id": 2, "level": "Warning", "message": "Hospital capacity reaching limits in North Zone.", "time": "1 hour ago"},
    ]
    return render_template('authority.html', stats=stats, alerts=alerts)

@app.route('/citizen')
def citizen():
    # Mock data for citizen dashboard
    user_status = {
        "risk_score": 15, # 0-100
        "status": "Safe",
        "last_updated": "Just now"
    }
    recommendations = [
        "Wear a mask in crowded areas.",
        "Maintain social distancing.",
        "Wash hands frequently."
    ]
    return render_template('citizen.html', status=user_status, recommendations=recommendations)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
