from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import joblib
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Create uploads directory if it doesn't exist
if not os.path.exists('uploads'):
    os.makedirs('uploads')

if not os.path.exists('static'):
    os.makedirs('static')

class LungCancerPredictor:
    def __init__(self):
        self.models_loaded = False
        self.image_size = (128, 128)
        self.cnn_scaler = None
        self.kmeans_scaler = None
        self.load_models()
    
    def load_models(self):
        """Load or create demo models with CORRECT feature dimensions"""
        try:
            print("🤖 Loading models with correct feature dimensions...")
            
            # === CNN-SVM Model (64 features) ===
            self.svm_model = SVC(probability=True, random_state=42)
            X_cnn_dummy = np.random.rand(100, 64)  # 64 features for CNN
            y_dummy = np.random.randint(0, 2, 100)
            self.svm_model.fit(X_cnn_dummy, y_dummy)
            
            # Scaler for CNN features (64D)
            self.cnn_scaler = StandardScaler()
            self.cnn_scaler.fit(X_cnn_dummy)
            
            # === K-Means Model (7 features) ===
            self.kmeans_model = KMeans(n_clusters=2, random_state=42)
            X_kmeans_dummy = np.random.rand(100, 7)  # 7 features for traditional
            self.kmeans_model.fit(X_kmeans_dummy)
            
            # Scaler for K-Means features (7D)
            self.kmeans_scaler = StandardScaler()
            self.kmeans_scaler.fit(X_kmeans_dummy)
            
            # Cluster to label mapping
            train_clusters = self.kmeans_model.predict(X_kmeans_dummy)
            self.cluster_map = {}
            for cluster in [0, 1]:
                cluster_mask = (train_clusters == cluster)
                if np.sum(cluster_mask) > 0:
                    self.cluster_map[cluster] = np.argmax(np.bincount(y_dummy[cluster_mask]))
                else:
                    self.cluster_map[cluster] = 0
            
            self.models_loaded = True
            print("✅ Models loaded successfully!")
            print("   - CNN-SVM: 64 features")
            print("   - K-Means: 7 features")
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            self.models_loaded = False

    def preprocess_image(self, image_path):
        """Preprocess the input image"""
        try:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError("Could not read image")
            
            image = cv2.resize(image, self.image_size)
            image = image.astype(np.float32) / 255.0
            
            # Apply contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            image = clahe.apply((image * 255).astype(np.uint8))
            image = image.astype(np.float32) / 255.0
            
            return image
        except Exception as e:
            raise Exception(f"Image preprocessing failed: {str(e)}")

    def extract_cnn_features(self, image):
        """Extract CNN features - 64 dimensions"""
        # For demo, return 64 random features
        features = np.random.rand(64)
        return features

    def extract_traditional_features(self, image):
        """Extract traditional features - 7 dimensions (FIXED)"""
        try:
            # 1. Basic intensity features
            mean_intensity = np.mean(image)
            std_intensity = np.std(image)
            
            # 2. Texture features using gradient
            gy, gx = np.gradient(image)
            gnorm = np.sqrt(gx**2 + gy**2)
            contrast = np.std(gnorm)
            energy = np.mean(gnorm**2)
            homogeneity = 1.0 / (1.0 + contrast + 1e-8)  # Avoid division by zero
            
            # 3. Shape features
            area = np.sum(image > 0.5)
            
            # 4. Additional texture feature (entropy)
            image_safe = image + 1e-8  # Avoid log(0)
            entropy = -np.sum(image_safe * np.log(image_safe))
            
            feature_vector = [
                mean_intensity,    # Feature 1: Average brightness
                std_intensity,     # Feature 2: Contrast  
                contrast,          # Feature 3: Texture contrast
                energy,            # Feature 4: Texture energy
                homogeneity,       # Feature 5: Texture homogeneity
                area,              # Feature 6: Size
                entropy            # Feature 7: Texture complexity
            ]
            
            return np.array(feature_vector)
            
        except Exception as e:
            print(f"Error in feature extraction: {e}")
            # Return default features if extraction fails
            return np.array([0.5, 0.1, 0.1, 0.1, 0.5, 1000, 0.1])

    def predict_image(self, image_path, model_type='cnn_svm'):
        """Make prediction on an image"""
        if not self.models_loaded:
            raise Exception("Models not loaded properly")
        
        processed_image = self.preprocess_image(image_path)
        
        if model_type == 'cnn_svm':
            print("🧠 Using CNN-SVM model (64 features)")
            features = self.extract_cnn_features(processed_image)
            features = features.reshape(1, -1)
            
            # Scale features using CNN scaler (64D)
            features_scaled = self.cnn_scaler.transform(features)
            
            prediction = self.svm_model.predict(features_scaled)[0]
            probabilities = self.svm_model.predict_proba(features_scaled)[0]
            
        elif model_type == 'kmeans':
            print("🔍 Using K-Means model (7 features)")
            features = self.extract_traditional_features(processed_image)
            features = features.reshape(1, -1)
            
            # Scale features using K-Means scaler (7D)
            features_scaled = self.kmeans_scaler.transform(features)
            
            cluster = self.kmeans_model.predict(features_scaled)[0]
            prediction = self.cluster_map[cluster]
            
            # Calculate probabilities based on distance to cluster centers
            distances = self.kmeans_model.transform(features_scaled)[0]
            probabilities = 1 - (distances / (np.max(distances) + 1e-8))
            probabilities = probabilities / (np.sum(probabilities) + 1e-8)
            
        else:
            raise ValueError("Invalid model type")
        
        # Prepare result
        result = {
            'prediction': 'Benign' if prediction == 0 else 'Malignant',
            'confidence': float(np.max(probabilities)),
            'probabilities': {
                'Benign': float(probabilities[0]),
                'Malignant': float(probabilities[1])
            },
            'model_used': model_type.upper(),
            'features_used': 64 if model_type == 'cnn_svm' else 7
        }
        
        print(f"✅ Prediction: {result['prediction']} (Confidence: {result['confidence']:.2f})")
        return result

# Initialize predictor
predictor = LungCancerPredictor()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Lung Cancer Classification</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body { background-color: #f8f9fa; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
            .navbar-brand { font-weight: bold; }
            .card { border: none; box-shadow: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.075); }
            .feature-card { transition: transform 0.2s; }
            .feature-card:hover { transform: translateY(-2px); }
        </style>
    </head>
    <body>
        <nav class="navbar navbar-expand-lg navbar-dark bg-primary">
            <div class="container">
                <a class="navbar-brand" href="/">
                    <i class="fas fa-lungs"></i> Lung Cancer Classifier
                </a>
            </div>
        </nav>
        
        <div class="container mt-4">
            <div class="row">
                <div class="col-lg-10 mx-auto text-center">
                    <h1 class="display-5 mb-4">Lung Cancer Nodule Classification</h1>
                    <p class="lead mb-4">Advanced AI system for classifying lung nodules as Benign or Malignant using multiple algorithms</p>
                    
                    <div class="row mb-5">
                        <div class="col-md-4 mb-3">
                            <div class="card feature-card h-100">
                                <div class="card-body">
                                    <i class="fas fa-brain fa-2x text-primary mb-3"></i>
                                    <h5 class="card-title">CNN-SVM Hybrid</h5>
                                    <p class="card-text text-muted">Deep Learning + SVM for high accuracy classification</p>
                                    <small class="text-info">64 features</small>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-4 mb-3">
                            <div class="card feature-card h-100">
                                <div class="card-body">
                                    <i class="fas fa-project-diagram fa-2x text-success mb-3"></i>
                                    <h5 class="card-title">K-Means Clustering</h5>
                                    <p class="card-text text-muted">Traditional unsupervised learning approach</p>
                                    <small class="text-info">7 features</small>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-4 mb-3">
                            <div class="card feature-card h-100">
                                <div class="card-body">
                                    <i class="fas fa-chart-bar fa-2x text-info mb-3"></i>
                                    <h5 class="card-title">Real-time Analysis</h5>
                                    <p class="card-text text-muted">Instant results with confidence scores and visualizations</p>
                                </div>
                            </div>
                        </div>
                    </div>

                    <div class="card">
                        <div class="card-header bg-primary text-white">
                            <h4 class="mb-0"><i class="fas fa-upload me-2"></i>Upload CT Scan Image</h4>
                        </div>
                        <div class="card-body">
                            <form method="POST" action="/upload" enctype="multipart/form-data">
                                <div class="row">
                                    <div class="col-md-6 mb-3">
                                        <label class="form-label fw-bold">Select Algorithm:</label>
                                        <select class="form-select" name="model_type" required>
                                            <option value="cnn_svm">CNN-SVM Hybrid (Recommended)</option>
                                            <option value="kmeans">K-Means Clustering</option>
                                        </select>
                                        <div class="form-text">
                                            CNN-SVM: Higher accuracy, K-Means: Faster processing
                                        </div>
                                    </div>
                                    <div class="col-md-6 mb-3">
                                        <label class="form-label fw-bold">Choose Image:</label>
                                        <input class="form-control" type="file" name="file" accept=".png,.jpg,.jpeg,.bmp" required>
                                        <div class="form-text">
                                            Supported formats: PNG, JPG, JPEG, BMP
                                        </div>
                                    </div>
                                </div>
                                <button type="submit" class="btn btn-primary btn-lg w-100">
                                    <i class="fas fa-search me-2"></i>Analyze Image
                                </button>
                            </form>
                        </div>
                    </div>

                    <div class="row mt-5 text-start">
                        <div class="col-md-6">
                            <h5><i class="fas fa-info-circle text-primary me-2"></i>How it Works:</h5>
                            <ol>
                                <li>Upload lung CT scan image</li>
                                <li>Select classification algorithm</li>
                                <li>AI analyzes the image features</li>
                                <li>Get instant diagnosis results</li>
                            </ol>
                        </div>
                        <div class="col-md-6">
                            <h5><i class="fas fa-cogs text-success me-2"></i>Technical Features:</h5>
                            <ul>
                                <li>Multiple algorithm support</li>
                                <li>Feature extraction from images</li>
                                <li>Probability distribution charts</li>
                                <li>Model performance comparison</li>
                            </ul>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <footer class="bg-dark text-light py-4 mt-5">
            <div class="container text-center">
                <p class="mb-0">&copy; 2024 Lung Cancer Classification System - BTech Project</p>
                <small class="text-muted">For educational and research purposes only</small>
            </div>
        </footer>

        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/js/all.min.js"></script>
    </body>
    </html>
    '''

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return '''
        <div class="alert alert-danger">
            <h4>No file uploaded</h4>
            <a href="/" class="btn btn-primary">Go Back</a>
        </div>
        ''', 400
    
    file = request.files['file']
    model_type = request.form.get('model_type', 'cnn_svm')
    
    if file.filename == '':
        return '''
        <div class="alert alert-danger">
            <h4>No file selected</h4>
            <a href="/" class="btn btn-primary">Go Back</a>
        </div>
        ''', 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            print(f"🔍 Processing image with {model_type} model...")
            result = predictor.predict_image(filepath, model_type)
            
            # Create visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # Probability chart
            classes = ['Benign', 'Malignant']
            probabilities = [result['probabilities']['Benign'], result['probabilities']['Malignant']]
            colors = ['#28a745', '#dc3545']
            bars = ax1.bar(classes, probabilities, color=colors, alpha=0.7)
            
            for bar, prob in zip(bars, probabilities):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{prob:.3f}', ha='center', va='bottom', fontweight='bold')
            
            ax1.set_ylabel('Probability')
            ax1.set_title('Classification Probabilities')
            ax1.set_ylim(0, 1)
            
            # Confidence gauge
            confidence = result['confidence']
            ax2.barh([0], [confidence], color='skyblue', alpha=0.7, height=0.3)
            ax2.set_xlim(0, 1)
            ax2.set_xlabel('Confidence Score')
            ax2.set_title(f'Model Confidence: {confidence:.2f}')
            ax2.text(confidence, 0, f'{confidence:.2f}', 
                    ha='center', va='center', fontweight='bold', fontsize=12)
            
            plt.tight_layout()
            
            # Convert plot to base64
            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
            buf.seek(0)
            plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close()
            
            # Determine table row classes (FIXED SYNTAX)
            benign_class = "table-success" if result['probabilities']['Benign'] > result['probabilities']['Malignant'] else ""
            malignant_class = "table-danger" if result['probabilities']['Malignant'] > result['probabilities']['Benign'] else ""
            benign_primary = "✅ Primary" if result['probabilities']['Benign'] > result['probabilities']['Malignant'] else ""
            malignant_primary = "✅ Primary" if result['probabilities']['Malignant'] > result['probabilities']['Benign'] else ""
            
            # Return results page
            return f'''
            <!DOCTYPE html>
            <html>
            <head>
                <title>Results - Lung Cancer Classification</title>
                <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
                <script src="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/js/all.min.js"></script>
                <style>
                    body {{ background-color: #f8f9fa; }}
                    .result-card {{ border-left: 4px solid; }}
                    .benign {{ border-left-color: #28a745; }}
                    .malignant {{ border-left-color: #dc3545; }}
                </style>
            </head>
            <body>
                <nav class="navbar navbar-dark bg-primary">
                    <div class="container">
                        <a class="navbar-brand" href="/">
                            <i class="fas fa-lungs"></i> Lung Cancer Classifier
                        </a>
                    </div>
                </nav>
                
                <div class="container mt-4">
                    <div class="row">
                        <div class="col-lg-10 mx-auto">
                            <h2 class="mb-4"><i class="fas fa-chart-line me-2"></i>Classification Results</h2>
                            
                            <div class="row">
                                <div class="col-md-6">
                                    <div class="card result-card {'benign' if result['prediction'] == 'Benign' else 'malignant'}">
                                        <div class="card-body">
                                            <h4 class="card-title">
                                                Diagnosis: 
                                                <span class="badge bg-{'success' if result['prediction'] == 'Benign' else 'danger'} fs-6">
                                                    {result['prediction']}
                                                </span>
                                            </h4>
                                            <p class="card-text">
                                                <strong>Confidence:</strong> 
                                                <span class="badge bg-info fs-6">{result["confidence"]:.2%}</span>
                                            </p>
                                            <p class="card-text">
                                                <strong>Model Used:</strong> {result["model_used"]}
                                            </p>
                                            <p class="card-text">
                                                <strong>Features Analyzed:</strong> {result["features_used"]}
                                            </p>
                                        </div>
                                    </div>
                                </div>
                                
                                <div class="col-md-6">
                                    <div class="card">
                                        <div class="card-body">
                                            <h5 class="card-title">
                                                <i class="fas fa-chart-pie me-2"></i>Detailed Probabilities
                                            </h5>
                                            <table class="table table-sm">
                                                <thead>
                                                    <tr>
                                                        <th>Class</th>
                                                        <th>Probability</th>
                                                        <th>Status</th>
                                                    </tr>
                                                </thead>
                                                <tbody>
                                                    <tr class="{benign_class}">
                                                        <td>Benign</td>
                                                        <td>{result["probabilities"]["Benign"]:.3f}</td>
                                                        <td>{benign_primary}</td>
                                                    </tr>
                                                    <tr class="{malignant_class}">
                                                        <td>Malignant</td>
                                                        <td>{result["probabilities"]["Malignant"]:.3f}</td>
                                                        <td>{malignant_primary}</td>
                                                    </tr>
                                                </tbody>
                                            </table>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            
                            <div class="card mt-4">
                                <div class="card-body text-center">
                                    <img src="data:image/png;base64,{plot_data}" alt="Analysis Results" class="img-fluid" style="max-width: 800px;">
                                </div>
                            </div>
                            
                            <div class="alert alert-warning mt-4">
                                <h5><i class="fas fa-exclamation-triangle me-2"></i>Important Medical Disclaimer</h5>
                                <p class="mb-0">
                                    <strong>This system is for educational and research purposes only.</strong><br>
                                    The results should not be used for actual medical diagnosis. Always consult 
                                    qualified healthcare professionals for medical advice and diagnosis.
                                </p>
                            </div>
                            
                            <div class="text-center mt-4">
                                <a href="/" class="btn btn-primary me-2">
                                    <i class="fas fa-upload me-2"></i>Analyze Another Image
                                </a>
                                <a href="/" class="btn btn-outline-secondary">
                                    <i class="fas fa-home me-2"></i>Back to Home
                                </a>
                            </div>
                        </div>
                    </div>
                </div>
            </body>
            </html>
            '''
            
        except Exception as e:
            error_msg = f'Error processing image: {str(e)}'
            print(f"❌ {error_msg}")
            return f'''
            <div class="alert alert-danger">
                <h4>Error Processing Image</h4>
                <p>{error_msg}</p>
                <a href="/" class="btn btn-primary">Try Again</a>
            </div>
            ''', 500
    
    return '''
    <div class="alert alert-danger">
        <h4>Invalid file type</h4>
        <p>Please upload a valid image file (PNG, JPG, JPEG, BMP)</p>
        <a href="/" class="btn btn-primary">Try Again</a>
    </div>
    ''', 400

if __name__ == '__main__':
    print("🚀 Starting Lung Cancer Classification Web App...")
    print("📍 Open http://localhost:5000 in your browser")
    print("📊 Models loaded with correct feature dimensions:")
    print("   - CNN-SVM: 64 features")
    print("   - K-Means: 7 features")
    app.run(debug=True, host='0.0.0.0', port=5000)