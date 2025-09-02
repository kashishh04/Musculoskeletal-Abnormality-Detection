# Musculoskeletal Abnormalities Detection Using Deep Learning

This project utilizes deep learning techniques to detect abnormalities in X-ray images of various body parts, including the wrist, finger, humerus, forearm, and hand. The application is developed using **Flask** and leverages TensorFlow/Keras for model inference, offering an intuitive web-based interface for medical professionals or researchers.

---

## 🚀 Features

- 🩻 **X-ray Image Analysis**: Upload X-ray images and receive predictions for abnormalities with high accuracy.
- 🧠 **Pre-trained Models**: Specialized deep learning models trained on datasets for detecting abnormalities in:
  - Wrist (XR_WRIST)
  - Finger (XR_FINGER)
  - Humerus (XR_HUMERUS)
  - Forearm (XR_FOREARM)
  - Hand (XR_HAND)
- 📊 **Confidence Score**: Each prediction includes a confidence level to gauge the reliability of the result.
- 💡 **User-Friendly Interface**: Simple and responsive design for easy interaction.
- 📄 **Detailed Results Page**: Provides predictions with accompanying visual and textual feedback.

---

## 🧰 Technologies Used

### 🔧 Backend:
- **Flask** – Lightweight Python web framework.
- **TensorFlow/Keras** – Deep learning frameworks for model inference.

### 🎨 Frontend:
- **HTML, CSS, Bootstrap** – For a responsive and modern UI.

### 🖼️ Image Processing:
- **scikit-image** – Used for preprocessing and transforming medical images.

## 🧠 Model Architecture

The model is built on **InceptionResNetV2**, a powerful CNN architecture pre-trained on ImageNet and fine-tuned for binary classification of musculoskeletal X-ray images using the MURA dataset.

**Architecture Summary:**

- **🔗 Base Model**: InceptionResNetV2 (excluding top layers)
- **🖼️ Input Shape**: (224, 224, 3)
- **🧱 Added Layers**:
  - Flatten layer
  - Dense (256 units, ReLU activation)
  - Dropout (rate: 0.5)
  - Output Dense (2 units, Softmax activation)
- **🧪 Optimizer**: RMSprop (learning rate = 0.0001)
- **🧮 Loss Function**: Categorical Crossentropy
- **📈 Training Methods**:
  - Data augmentation
  - Class weighting to handle imbalance
---

## 📥 Download Pretrained Models

All models used for inference in this project are hosted on Hugging Face:

👉 [**Download Model Weights Here**](https://huggingface.co/thor15/Musculoskeletal-Abnormalities-Detection-by-DL/tree/main)

> After downloading, place the model files into the `models/` directory within your project (create the directory if it doesn't exist).

---# Musculoskeletal--Abnormality--Detection

## 📄 Related Research

This project is based on methods and findings discussed in the following paper:

> 🔗 [**IEEE Paper: "An Efficient Hybrid Deep Learning Model for Detecting Musculoskeletal Abnormalities"**]([https://ieeexplore.ieee.org/document/10927966])(https://ieeexplore.ieee.org/document/10927835))

