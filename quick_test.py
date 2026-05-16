import os
import cv2
import numpy as np
from generate_test_samples import create_benign_nodule, create_malignant_nodule

def quick_model_test():
    """Quick test of the model with generated samples"""
    print("🧪 Quick Model Testing...")
    
    # Create a few test samples in memory
    test_samples = []
    
    # Generate 2 benign and 2 malignant samples
    for i in range(2):
        benign_sample = create_benign_nodule()
        malignant_sample = create_malignant_nodule()
        
        test_samples.append(('benign', benign_sample))
        test_samples.append(('malignant', malignant_sample))
    
    print(f"✅ Generated {len(test_samples)} test samples")
    
    # Simulate model predictions (replace with actual model calls)
    for true_label, image in test_samples:
        # Simulate model prediction (replace this with actual model prediction)
        if np.random.random() > 0.2:  # 80% accuracy simulation
            predicted_label = true_label
            confidence = np.random.uniform(0.7, 0.95)
        else:
            predicted_label = 'malignant' if true_label == 'benign' else 'benign'
            confidence = np.random.uniform(0.3, 0.6)
        
        print(f"📊 Sample: True={true_label:9} | Predicted={predicted_label:9} | Confidence={confidence:.2f} | {'✅' if true_label == predicted_label else '❌'}")
    
    print("\n🎯 Quick test completed!")

if __name__ == "__main__":
    quick_model_test()