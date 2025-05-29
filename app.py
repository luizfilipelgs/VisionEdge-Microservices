import os
from datetime import datetime
from flask import Flask, render_template, request, jsonify, Response
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from detector.video_processor import VideoProcessor
from detector.event_logger import EventLogger
from detector.business_analytics import BusinessAnalytics
import logging
import json
import time

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

# Ensure required directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('events', exist_ok=True)

# Initialize video processor and event logger
try:
    video_processor = VideoProcessor(business_type='supermarket')
    event_logger = EventLogger('events/detection_events.txt', business_type=video_processor.business_analytics.business_type)
    logger.info("Sistema inicializado com sucesso!")
except Exception as e:
    logger.error(f"Erro ao inicializar o sistema: {str(e)}")
    raise

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Erro ao renderizar página inicial: {str(e)}")
        return "Erro ao carregar a página", 500

@app.route('/set_business_type', methods=['POST'])
def set_business_type():
    try:
        data = request.get_json()
        business_type = data.get('business_type')
        if not business_type:
            return jsonify({'error': 'Tipo de negócio não fornecido'}), 400
        video_processor.business_analytics.business_type = business_type
        video_processor.business_analytics.reset_metrics()
        # Atualizar o event_logger para novo tipo de negócio
        global event_logger
        event_logger = EventLogger('events/detection_events.txt', business_type=business_type)
        return jsonify({'message': f'Tipo de negócio alterado para {business_type}'})
    except Exception as e:
        logger.error(f"Erro ao alterar tipo de negócio: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/get_business_insights')
def get_business_insights():
    try:
        return jsonify(video_processor.get_business_insights())
    except Exception as e:
        logger.error(f"Erro ao obter insights: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'Nenhum arquivo enviado'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Nenhum arquivo selecionado'}), 400

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Inicia a detecção com o arquivo salvo
        if video_processor.connect(filepath):
            return jsonify({'message': 'Arquivo enviado e processamento iniciado'})
        else:
            return jsonify({'error': 'Erro ao processar o arquivo'}), 500

@app.route('/start_detection', methods=['POST'])
def start_detection():
    try:
        data = request.get_json()
        source = data.get('source')
        
        if not source:
            return jsonify({'error': 'URL do stream não fornecida'}), 400

        if video_processor.connect(source):
            return jsonify({'message': 'Detecção iniciada com sucesso'})
        else:
            return jsonify({'error': 'Erro ao conectar à fonte de vídeo'}), 500

    except Exception as e:
        logger.error(f"Erro ao iniciar detecção: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/stop_detection', methods=['POST'])
def stop_detection():
    try:
        video_processor.disconnect()
        return jsonify({'message': 'Detecção parada com sucesso'})
    except Exception as e:
        logger.error(f"Erro ao parar detecção: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/video_feed')
def video_feed():
    try:
        logger.info("Iniciando stream de vídeo")
        return Response(video_processor.generate_frames(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        logger.error(f"Erro na rota de video_feed: {str(e)}")
        return "Erro ao gerar stream de vídeo", 500

@app.route('/get_insights')
def get_insights():
    try:
        if not video_processor.is_running:
            return jsonify({'error': 'Stream não está ativo'}), 400
        
        metrics = video_processor.get_metrics()
        return jsonify(metrics)
    except Exception as e:
        logger.error(f"Erro ao obter insights: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/get_events')
def get_events():
    try:
        events = event_logger.get_events()
        return jsonify({'events': events})
    except Exception as e:
        logger.error(f"Erro ao obter eventos: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    """Handler para erro 404"""
    return jsonify({'error': 'Recurso não encontrado'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handler para erro 500"""
    return jsonify({'error': 'Erro interno do servidor'}), 500

if __name__ == '__main__':
    logger.info("Iniciando servidor Flask...")
    try:
        app.run(debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        logger.error(f"Erro ao iniciar o servidor: {str(e)}") 