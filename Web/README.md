# Ink Scribe AI

Build a full-stack Web Application for a "Handwritten Recognition System (HNRS)" with a modern, responsive, and clean AI-themed UI/UX.

---

### 1. 📂 CURRENT PROJECT STRUCTURE CONTEXT

The current root repository (`GROUP1_THEMEB_HANDWRITTENNUMBERRE...`) contains the following existing folders:

├── Datasets/              # Contains Train/Test CSV data (Numbers & EngText)

├── Documents/             # Project reports, requirements, and meeting minutes

├── Figures/               # EDA plots and model training graphs

├── Models/                # Trained ML/DL models:

│   ├── digit_cnn_model.keras   # Keras CNN model for digits (0-9)

│   ├── best_model.keras        # Alternative Keras model

│   ├── letter_customcnn.pth    # PyTorch Custom CNN model for letters (A-Z)

│   └── best_crnn_model.pt      # PyTorch CRNN model for sequence/text recognition

└── Notebook/              # Jupyter notebooks for training and evaluation

---

### 2. 🏗️ NEW WEBSITE FOLDER STRUCTURE REQUIREMENT

Create a new root directory named `website` with a full-stack architecture including a dedicated database folder:

website/

├── frontend/             # React application (runs with: npm run dev)

│   ├── src/

│   │   ├── components/   # Canvas, Header, HistoryTable, PredictionCard

│   │   ├── pages/        # Dashboard, ModelEvaluation, History

│   │   └── services/     # API integration services

│   └── package.json

└── backend/              # Node.js / Express backend server (runs with: npm run / npm start)

    ├── database/         # 🗄️ Database storage & ORM/Migration files

    │   ├── db.sqlite     # SQLite database file

    │   ├── schema.sql    # Database schema definitions

    │   └── dbClient.js   # Database connection initialization

    ├── models_bridge/    # Python execution scripts to call PyTorch (.pt/.pth) and Keras (.keras) models

    ├── controllers/      # API logic for predictions and history management

    ├── routes/           # Express endpoint routes

    ├── server.js

    └── package.json

---

### 3. 📄 HEADER & CORE DESCRIPTION REQUIREMENT

At the top of the main dashboard UI, strictly keep the exact header text:

"This Handwritten Recognition System is an advanced AI-powered platform utilizing custom Convolutional Neural Networks (CNN) to accurately classify, read, and interpret isolated handwritten digits (0-9) and uppercase English characters (A-Z) in real-time from various input sources."

---

### 4. 📚 ACADEMIC PROJECT REQUIREMENTS CONTEXT (COS30018 - HNRS)

Ensure the application fulfills all project requirements outlined in the course specifications:

- **Task 1: Image Preprocessing Visualizer**: 

  - Showcase real-time step-by-step image transformation: Original Image -> Grayscale -> Binarization/Thresholding -> Resized (28x28 normalized pixel tensor).

- **Task 2: Image Segmentation & Multi-Digit/Text Handling**:

  - Automatically segment multi-digit numbers or words into bounding boxes for individual character recognition.

- **Task 3: Multi-Model Inference & Selection**:

  - Support model switching between Digit CNN (Keras) and English Text models (Custom CNN & CRNN PyTorch).

- **Task 4: Evaluation & Analytics Dashboard**:

  - Display confidence scores, inference latency (ms), class probabilities bar chart, and confusion matrix preview.

---

### 5. 🧠 BACKEND WORKFLOW & ML MODEL INTEGRATION

The backend must route requests to call the corresponding trained model weights located in `../Models/`:

- **Digit Prediction Route (`/api/predict/digit`)**:

  - Uses `../Models/digit_cnn_model.keras` (or `best_model.keras`) to predict digits (0-9).

- **English Text Recognition Route (`/api/predict/text`)**:

  - Uses `../Models/best_crnn_model.pt` for full sequence text or `../Models/letter_customcnn.pth` for single letters (A-Z).

- **Workflow**:

  1. Frontend sends Base64 canvas image / uploaded image file to API.

  2. Backend preprocesses image to 28x28 grayscale matrix.

  3. Python bridge script executes inference using PyTorch / TensorFlow.

  4. Result is saved to the SQLite database and returned to the UI.

---

### 6. 🗄️ DATABASE & PREDICTION HISTORY TABLE

Use the `website/backend/database/` directory to manage a persistent SQLite/PostgreSQL database.

- **Database Schema (`prediction_history`)**:

  - `id`: INTEGER PRIMARY KEY AUTOINCREMENT

  - `created_at`: TIMESTAMP (Default CURRENT_TIMESTAMP)

  - `input_type`: TEXT ('Interactive Canvas', 'File Upload', 'CSV Test File')

  - `model_used`: TEXT ('digit_cnn_model.keras', 'best_crnn_model.pt', 'letter_customcnn.pth')

  - `predicted_text`: TEXT

  - `confidence_score`: FLOAT

  - `execution_time_ms`: INTEGER

  - `image_data_url`: TEXT (Base64 thumbnail preview)

- **UI History Component**:

  - Positioned directly below the Header/Control Panel.

  - Interactive table displaying prediction history with: `ID`, `Timestamp`, `Input Thumbnail`, `Model Used`, `Predicted Output`, `Confidence`, and `Actions (Delete/View Details)`.

  - Includes Search by character, Filter by Model, and Pagination.

MUst have function camera capture

This project was built with [Lovable](https://lovable.dev).

**Live app**: https://pen-to-pixel-ai.lovable.app

## Build with Lovable

Continue developing this project in the [Lovable editor](https://lovable.dev/projects/048d62ca-6467-43ea-b9da-9bfd713ef01e).

- **Ship faster**: describe what you want to build and Lovable handles the code.
- **Stay in sync**: every change made in Lovable is committed straight to this repository.
- **Full ownership**: this code is yours. Push to `main` on GitHub and your changes sync back into Lovable, ready for your next prompt.

## Development

Prefer working locally? You need Node.js and npm — [install with nvm](https://github.com/nvm-sh/nvm#installing-and-updating).

```sh
git clone <this-repository-url>
cd <repository-name>
npm i
npm run dev
```
