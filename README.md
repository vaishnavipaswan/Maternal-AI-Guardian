
# MAG👶 

MAG is an AI-powered mobile and web application designed to provide pregnancy risk assessment, cultural care guidance, emergency support, and medical assistance for expecting mothers and families—offline-first and multilingual.

## 🔥 Built for Hackathons
- ⏱ 12-hour MVP built with scalability in mind
- 📱 React Native frontend (Expo/CLI)
- 🧠 TensorFlow.js model for on-device predictions
- 🌐 Localized HTML version for rapid access
- 🆘 Emergency SOS and offline-first profiles

---

## 📦 Features

| Feature                           | Description                                                                 |
|-----------------------------------|-----------------------------------------------------------------------------|
| 💡 Risk Prediction                | Predicts high/low pregnancy health risks using 16 features & ML model      |
| 📆 Predictive Timeline            | Week-by-week pregnancy care reminders                                      |
| 📚 Pregnancy Books                | Educational PDFs for both parents                                           |
| 🧑‍⚕️ Doctor Assistance            | Find nearby gynecologists by location                                      |
| 🧘 Cultural & Regional Care       | Adjusted diet, herbs, and fasting advice based on region (e.g., Maharashtra)|
| 🌧 Weather & Festival Alerts      | Contextual health tips for monsoon, summer, and fasts like Navratri        |
| 📱 Offline-First Accounts         | Mother and Father separate profiles                                        |
| 🚨 Emergency SOS Button           | Quick access to ambulance, doctor, husband                                 |
| 🎮 Gamification                   | Points & achievements for following advice                                 |

---

## 🧠 Machine Learning Model

- **Architecture**: 3-layer Dense Neural Network (16 → 64 → 32 → 1)
- **Input Features**: 
  - Age, Heart Rate, Body Temp, Blood Sugar, BP (Sys/Dia)
  - Swelling, Nausea, Pain, Bleeding, Stress, Depression
  - Sleep, Diabetes, Anemia, Prior Miscarriage
- **Model Type**: Binary Classification (Risk: Yes/No)
- **Format**: `model.json` (TensorFlow.js)
- **Loss**: Binary Crossentropy
- **Optimizer**: Adam (lr=0.001)
- **Accuracy**: ~87% on validation set of 1000 entries

---

## 🛠 Prerequisites

Make sure to have these installed before running:

```bash
# Install Node + Expo CLI (React Native)
npm install -g expo-cli

# Or use React Native CLI (for advanced builds)
npx react-native init MAGApp
```

### 📁 Project Structure (Frontend)

```
/
├── assets/
│   ├── model/
│   │   ├── model.json
│   │   └── group1-shard1of1.bin
│   └── logo.png
├── src/
│   ├── screens/
│   ├── utils/
│   │   └── predict.ts
│   └── App.tsx
├── MAG.html            # Web prototype (offline-friendly)
└── README.md
```

---

## 🚀 Run Instructions

### 🔧 For Web (Local HTML version)
Open `MAG.html` in any modern browser with internet disabled (model runs offline!).

### 📱 For Mobile (Expo)
```bash
cd MAGApp
npm install
npx expo start
```

Scan QR with Expo Go App.

---

## 🔐 Security & Ethics

- ✅ No data leaves the device
- ✅ Only client-side prediction using TF.js
- ⚠️ Not a replacement for professional diagnosis
- 🔐 Emergency contacts are stored only locally

---

## 🧾 References

- 📘 Model trained on anonymized dataset (1000 entries, 16 features)
- 🧬 ML pipeline: Keras → TensorFlow.js (`model.json`)
- 🌍 Cultural adjustments by consulting regional healthcare guides
- 🩺 Doctor assistance simulated with mock data

---

## 💡 Judges: Why This Matters

> 83% of Indian women in rural areas lack access to maternity-specific care.  
> Our app empowers users with **offline**, **language-specific**, **AI-powered**, **hyperlocal health predictions** in under 5 seconds.

---

## 🔗 Useful Commands

```bash
# Clean install
rm -rf node_modules && npm install

# Debug Android
npx react-native run-android

# Serve HTML (optional)
npx http-server ./
```

---

## 🧰 Tech Stack

- Frontend: HTML5, CSS3, React Native, TypeScript
- Backend: None (offline-first)
- AI: TensorFlow.js, Keras (converted model)
- Dev Tools: Expo, VSCode, Git, Android Emulator

---

## 🧑‍💻 Team

- 👩‍💻 Sahas Sujit Mokashi - ML & Fullstack Dev
- 👩‍⚕️ Domain expert - Public health research
- 🎨 UI/UX - Visual branding + gamification

---

## 📃 License

MIT License – Open for community improvement.

---

## ✅ Hash Verification (Optional Security)

Verify your model file:
```bash
# Run this in project root
sha256sum assets/model/model.json
sha256sum assets/model/group1-shard1of1.bin
```
