# Smart Waste Segregation System

A deep learning and IoT-powered project that classifies and segregates household waste (plastic, metal, glass, organic) using image classification with CNNs and real-time actuation through Raspberry Pi & ESP32.

---

## Project Overview

This project demonstrates a **Smart Waste Bin** capable of automatically sorting waste into the correct bin. It leverages:
- **Deep Learning (CNN)** for image-based waste classification.
- **Sound analysis** to aid classification.
- **ESP32** for IoT control & mechanical actuation.
- **PyQt6 desktop interface** for real-time monitoring & visualization.

The solution addresses the global challenge of improper waste disposal by ensuring recyclables and organics are automatically separated at the point of disposal.

---

## Algorithm Used

This project uses a **Supervised Deep Learning** algorithm:

- **Model Type**: Convolutional Neural Network (CNN)
- **Learning Paradigm**: Supervised Learning (images are labeled by category)
- **Training Method**: Backpropagation with Gradient Descent
- **Optimizer**: Adam (Adaptive Gradient Descent)
- **Loss Function**: Cross-Entropy Loss (multi-class classification)
- **Data Handling**: Mini-batch training with DataLoader

CNNs are chosen over traditional ML because they automatically learn features (edges, shapes, textures) from raw images.

---

## Project Structure

```bash
Smart-waste-segregation/
│
├── dataset/                  # Waste images dataset (plastic, glass, metal, organic)
├── train_model.py            # CNN training script
├── waste_model.pt            # Trained PyTorch model (saved weights)
├── ui_app.py                 # PyQt6 interface for live webcam classification
├── esp32_code/               # ESP32 firmware for tray & bin control
├── raspberry_pi_code/        # Pi-side integration with ML + control logic
└── README.md                 # This documentation
```

---

## Features

- **Image-based waste classification** using CNN
- **One-time task reminders** for disposal & monitoring
- **ESP32-controlled tray rotation** to drop waste into the correct bin
- **Real-time PyQt6 dashboard** with webcam feed & classification summary
- **Multi-modal classification** (future: sound-based analysis)
- **IoT Integration** between Raspberry Pi and ESP32

---

## Hardware Requirements

- ESP32 (WiFi-enabled)
- Servo/Stepper Motor (for rotating top tray)
- 4 Waste Bins (plastic, organic, glass, metal)
- Camera Module (USB/Webcam or Pi Camera)
- Power Supply & Driver Circuitry

---

## Software Requirements

- **Python 3.10+**
- **PyTorch** (deep learning framework)
- **OpenCV** (image capture & preprocessing)
- **PyQt6** (UI dashboard)
- **SoundDevice + NumPy** (future sound classification)
- **Arduino/ESP-IDF** for ESP32 firmware

Install dependencies:
```bash
pip install torch torchvision opencv-python pyqt6 sounddevice numpy tensorflow
```

---

## Model Training

1. Place the dataset under `dataset/` with subfolders for each class:
   ```bash
   dataset/
   ├── plastic/
   ├── glass/
   ├── metal/
   └── organic/
   ```

2. Run the training script:
   ```bash
   python train_model.py
   ```

3. Model will be saved as `waste_model.pt`.

---

## Running the System

1. Start the PyQt6 UI:
   ```bash
   python ui_app.py
   ```
   - Displays live webcam feed.
   - Shows predicted waste type.

2. The prediction is sent to ESP32:
   - ESP32 receives bin ID (1=Plastic, 2=Glass, 3=Metal, 4=Organic).
   - Motor rotates the tray to drop waste into the corresponding bin.

---

## Example Workflow

1. User drops waste item onto the tray.
2. Camera captures image → sent to CNN.
3. CNN predicts category (e.g., **Plastic**).
4. Then the prediction is sent to the ESP32 for actuation of the servo.
5. Tray rotates and directs waste to the correct bin.
6. UI updates with classification history.

---

## Future Improvements

- **Sound-based classification** for metallic items (can detect)
- **Cloud integration** for data logging & monitoring (future extension)
- **Mobile app** to show live stats & waste analytics (future extension)
- **Optimization** with lightweight CNNs (MobileNet, EfficientNet) for faster inference
- 
---

## Team
- Dineshkumar D, 
- A big thank you to my amazing teammates, Aswin S, Aravind S, and Abdullah, for making this project a success through teamwork and dedication.
 
---

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

<img width="839" height="637" alt="SWG-CAD" src="https://github.com/user-attachments/assets/c9494d9a-8ac8-4cb2-a5e9-a27d375f54e2" />


https://github.com/user-attachments/assets/6a686e42-7809-4980-9101-69b41ec1c727


https://github.com/user-attachments/assets/8dd4254c-1f23-4664-9159-0d1b3c360bbe

