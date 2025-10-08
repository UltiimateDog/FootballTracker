# Computer Vision System for Automatic Ball Detection and Speed Estimation in Football Matches

**Alan Perez Hernandez – 177730**
**September 2025**

---

## 1. Introduction

Football is one of the most followed sports worldwide, and the FIFA World Cup 2026 will attract millions of viewers. Beyond entertainment, the analysis of game dynamics is increasingly important for refereeing, broadcasting, and player performance evaluation.

Traditionally, accurate ball-tracking systems rely on expensive infrastructure, such as multiple synchronized cameras or sensor-embedded balls. These solutions, however, are not widely accessible.

This project proposes a low-cost, computer vision–based solution that detects the football in match videos, tracks its position, and estimates its speed. The approach uses **OpenCV** for image preprocessing and tracking, and **PyTorch** for deep learning–based object detection.

Additionally, an **iOS mobile application** will be developed to connect to a central server running the computer vision models. This app will allow users to upload or stream match footage, view ball trajectories, and receive real-time speed feedback, thus significantly enhancing accessibility.

---

## 2. Problem Identification

The ball is small relative to the field and often occluded by players. Lighting conditions, crowd movement, and motion blur further complicate detection. Thus, the engineering problem involves:

* Detecting a small, fast-moving object under varying conditions.
* Ensuring robustness against occlusion and clutter.
* Converting pixel measurements into real-world distances.
* Estimating ball velocity with reasonable accuracy.
* Designing a user-friendly interface (mobile app) for accessibility.

---

## 3. Proposed Solution

The proposed system follows a four-step pipeline:

### 3.1 Preprocessing

Video frames are first processed to reduce noise and enhance contrast. **Gaussian filters** and **edge detection** are applied to improve object boundaries.

### 3.2 Ball Detection

Two complementary approaches are considered:

* **Classical Computer Vision**: Use color filtering (if the ball contrasts with the field) and **Circular Hough Transform** to detect circular shapes.
* **Deep Learning**: Train or fine-tune a **YOLO (You Only Look Once)** model in **PyTorch** to detect the football. This approach is more robust against occlusion and lighting variation.

### 3.3 Tracking

Once detected, the ball is tracked across frames using a **Kalman Filter** or **centroid tracking**. This ensures continuity even when the ball is partially occluded.

### 3.4 Speed Estimation

Let ((x_t, y_t)) be the ball coordinates in frame (t). The displacement is computed as:

```math
d = \sqrt{(x_{t+1} - x_t)^2 + (y_{t+1} - y_t)^2}
```

Pixels are converted to meters using known field dimensions (e.g., pitch width of 68 meters). The velocity is then:

```math
v = \frac{d}{\Delta t}
```

Where (\Delta t) is the time between frames.

---

## 4. Accessibility Considerations

To enhance accessibility:

* The system can run on consumer-grade laptops with a single camera.
* Results (e.g., ball speed) can be displayed as **text overlays** or provided via **audio output** for visually impaired users.
* An **iOS app** provides a simple interface for users to upload or stream video to the server, which runs the CV models and returns analytics in real time.
* The software will be **open-source**, reducing barriers for communities and smaller clubs.

---

## 5. System Constraints

For reliable operation, the system requires:

* A **static camera**: the recording device must remain fixed to avoid false motion detections.
* **Clear visibility** of the field: heavy occlusion or poor lighting reduces accuracy.
* **Standardized field dimensions**: necessary for correct pixel-to-meter calibration.
* **Sufficient frame rate** (at least 30 fps) to ensure accurate speed estimation.
* **Reliable network connection** for the iOS app to communicate with the server in real time.

---

## 6. Ethical Concerns

The system must respect ethical boundaries:

* The algorithm is designed **strictly for ball detection**, avoiding unnecessary tracking of players or spectators.
* Collected video data should **not be misused** for surveillance purposes.
* **Transparency** is ensured by releasing the methodology and code publicly.

---

## 7. Technical Resources

* **Programming Language**: Python (server) and Swift (iOS app).
* **Libraries**: OpenCV (image processing), PyTorch (deep learning), NumPy/SciPy (math operations), Matplotlib (visualization).
* **Hardware**: Standard webcam or smartphone camera.
* **Dataset**: Publicly available football match recordings for model training and evaluation.
* **Deployment**: Central server with CV models, connected to the iOS app client.

---

## 8. Evaluation Criteria

The solution will be evaluated based on:

* **Accuracy**: Precision and recall in ball detection.
* **Robustness**: Performance under occlusion, lighting changes, and motion blur.
* **Efficiency**: Real-time or near real-time execution.
* **Accessibility**: Usability on low-cost devices and mobile platforms.
* **Constraints**: System performs correctly under specified operational conditions.
* **Ethics**: Compliance with privacy and transparency principles.

---

## 9. Conclusion

This project proposes an accessible and ethical computer vision system for detecting and tracking a football in match recordings, estimating its speed using physics-based principles.

By combining classical image processing (**OpenCV**) with deep learning (**PyTorch**), the system balances simplicity and robustness. The inclusion of an **iOS application** enhances usability by enabling real-time interaction with the server-based CV models.

Such a solution could enhance **fan engagement** at the World Cup 2026, while also being usable in **community-level football analysis**.
