# Traffic-Sign-Classification-Using-Convolutional-Neural-Networks-CNN-
Traffic sign classification using CNN and PyTorch model
Traffic Sign Classification using CNN
Road Sign Recognition for Autonomous Systems
Prathmesh Giram
PG-DBDA, CDAC Kharghar
Abstract
This project presents a Convolutional Neural Network (CNN)-based system for classifying
German road signs using the GTSRB dataset. It also features a user-friendly web applica-
tion for real-time prediction and supports applications in autonomous vehicles and traffic
management systems.
1 Introduction
Traffic signs play a crucial role in road safety and autonomous driving. Recognizing these
signs accurately is essential for developing intelligent transportation systems. This project
leverages deep learning, particularly CNNs, to classify 43 types of traffic signs from images.
2 Dataset
Dataset: German Traffic Sign Recognition Benchmark (GTSRB)
• 43 traffic sign classes
• RGB images resized to 64x64 pixels
• Data normalized using ImageNet statistics
3 Model Architecture
Model: Custom CNN built using PyTorch
• Input: 64x64 RGB image
• Layers:
– Conv2D (3→32) → ReLU → MaxPool
– Conv2D (32→64) → ReLU → MaxPool
– Conv2D (64→128) → ReLU → MaxPool
1
Prathmesh Giram Traffic Sign Classifier
– Flatten → Linear(8192→256) → ReLU → Dropout → Linear(256→43)
• Loss Function: CrossEntropyLoss
• Optimizer: Adam
4 Environment Setup
Conda Environment: roadsign-env
Listing 1: Conda Environment Setup
name : r o a d s i g n −env
c h a n n e l s :
− d e f a u l t s
− p yt o rc h
− n v i d i a
d e p e n d e n c i e s :
− python =3.10
− p yt o rc h
− t o r c h v i s i o n
− t o r c h a u d i o
− pytorch−cuda =11.8
− pip
− pip :
− opencv−python
− m a t p l o t l i b
− tqdm
− pandas
− s c i k i t −l e a r n
− u l t r a l y t i c s
− p y t t s x 3
5 Training and Evaluation
• Batch Size: 32
• Epochs: 5
• Learning Rate: 0.001
• Accuracy Achieved: (Insert % here after evaluation)
• Model Saved At: outputs/best model.pth
2
Prathmesh Giram Traffic Sign Classifier
6 Web Application
Built Using: Streamlit
• Upload traffic sign images
• Real-time classification with confidence score
• Displays class name and prediction
• (Note: UI screenshot not shown here)
7 Project Structure
• app.py – Web application logic
• labels.py – Class names
• outputs/best model.pth – Trained model
• src/model.py – CNN model definition
• src/train.py – Model training script
• src/loaders.py – Data loaders and preprocessing
• src/eval.py – Evaluation script for accuracy and loss
• src/utils.py – Utility functions (e.g., plotting, logging)
8 Future Work
• Add Grad-CAM visualization
• Enable real-time webcam input
• Enhance performance with data augmentation
• Deploy on cloud (AWS/HuggingFace)
Acknowledgment
Special thanks to the GTSRB dataset creators and open-source communities of PyTorch and
Streamlit.
Built with by Prathmesh Giram
