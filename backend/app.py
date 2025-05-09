from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import subprocess
import os
import sys

app = Flask(__name__)
CORS(app)

@app.route('/start-analysis', methods=['POST'])
def run_start_analysis():
    print("Triggered /start-analysis")
    try:
        subprocess.Popen([sys.executable, 'analyzer/a2.py'], cwd=os.path.dirname(__file__))
        return jsonify({'message': 'Start analysis script launched!'})
    except Exception as e:
        print(f"Error starting analysis: {e}")
        return jsonify({'message': 'Failed to start analysis.'}), 500

@app.route('/get-insights', methods=['POST'])
def run_insights_analysis():
    print("Triggered /get-insights")
    try:
        subprocess.Popen([sys.executable, 'abc.py', '-q'], cwd=os.path.dirname(__file__))
        return jsonify({'message': 'Insights analysis script launched!'})
    except Exception as e:
        print(f"Error running insights: {e}")
        return jsonify({'message': 'Failed to get insights.'}), 500

@app.route('/get-report', methods=['POST'])
def run_report():
    print("Triggered /get-report")
    try:
        subprocess.Popen(
            [sys.executable, 'test3.py', r'D:\VisionFB\backend\behavior_logs\behavior_log_20250509_062848.csv'],
            cwd=os.path.dirname(__file__)
        )
        return jsonify({'message': 'Report script launched!'})
    except Exception as e:
        print(f"Error running report: {e}")
        return jsonify({'message': 'Failed to get report.'}), 500

@app.route('/latest-attendance')
def serve_attendance():
    print("Serving Attendance.csv")
    try:
        return send_file("Attendance.csv", mimetype='text/csv')
    except Exception as e:
        print(f"Error serving attendance file: {e}")
        return str(e), 500

@app.route('/stop-analysis', methods=['POST'])
def stop_analysis_mock():
    print("Triggered /stop-analysis")
    try:
        return jsonify({'message': 'Analysis stopped!'})
    except Exception as e:
        return jsonify({'message': 'Failed to stop analysis.'}), 500

if __name__ == '__main__':
    print("Flask app starting...")
    app.run(debug=True)
