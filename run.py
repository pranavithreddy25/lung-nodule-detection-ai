from app import create_app
import os

app = create_app()

if __name__ == '__main__':
    # Create necessary directories
    directories = ['uploads', 'app/static/images', 'trained_models']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✅ Created directory: {directory}")
    
    print("🚀 Starting Lung Cancer Classification Web App...")
    print("📍 Open http://localhost:2345 in your browser")
    app.run(debug=True, host='0.0.0.0', port=2345)