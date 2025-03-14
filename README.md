# spadeAI
This server acts as the **eyes** of the [Spade](https://github.com/spade-poker) AI Poker Dealer, specializing in real-time card detection. It uses computer vision and machine learning to identify community cards (via an overhead camera) and player cards (via smartphone images). 

## ✨ Features  
### 🌐 Community Card Detection (`getCommunityCards()`)  
- **Overhead Camera Integration**: Automatically captures tabletop video feed.  
- **Computer Vision Magic**: Uses color thresholding, contour detection, and perspective transforms to isolate cards.  
- **Rank & Suit Recognition**: Accurately detects card values (e.g., `A♠`, `K♥`).  

### 📱 Player Card Detection (`getPlayerCards()`)  
- **YOLOv8 Model Inference**: Lightning-fast card detection from player-submitted images.  
- **Mobile-Friendly**: Processes images from smartphones in real-time.  
- **Top-2 Results**: Returns the two highest-confidence detected cards.  

---

## 🛠️ Installation  
1. Clone the repository:  
   ```bash
   git clone https://github.com/your-username/spade-card-detection.git
   cd spade-card-detection
   ```
2. Create venv and install Requirements:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
