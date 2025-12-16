# Chinese Name Generator

A sophisticated full-stack web application that generates meaningful, phonetically accurate, and culturally appropriate Chinese names for non-Chinese speakers.

## Features

-   **Semantic Matching**: Uses `paraphrase-multilingual-MiniLM-L12-v2` to match your personal description with Chinese character meanings.
-   **Phonetic Alignment**: Converts English names to IPA and calculates Manhattan distance to finding sounding-alike characters.
-   **Demographic Scoring**: Ensures gender and age appropriateness using real-world statistical data.
-   **Tone Analysis**: Filters for euphonic 3-character tone combinations.
-   **Advanced Controls**: Fine-tune weights for Meaning, Sound, Gender, Age, and Creativity.

## Tech Stack

-   **Frontend**: Next.js, React, TypeScript (Dark UI).
-   **Backend**: Python Flask, Sentence-Transformers, Phonemizer, Pandas.

## Prerequisites

1.  **Python 3.10+**
2.  **Node.js 18+**
3.  **espeak-ng** (Required for phonetic analysis)
    -   *Windows*: [Download Installer](https://github.com/espeak-ng/espeak-ng/releases) or use standard `espeak`.
    -   *Linux*: `sudo apt-get install espeak-ng`
    -   *Mac*: `brew install espeak`

## Running Locally

### 1. Backend (Flask)

Navigate to the `backend` directory:

```bash
cd backend
```

Create a virtual environment (optional but recommended):

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the API server:

```bash
python generate_names.py
```
*The server will start on `http://localhost:5000`.*

### 2. Frontend (Next.js)

Open a new terminal and navigate to the `frontend` directory:

```bash
cd frontend
```

Install dependencies:

```bash
npm install
```

Run the development server:

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

## Deployment

The project is Dockerized for deployment (e.g., on Railway):
-   `backend/Dockerfile`: CPU-optimized PyTorch build.
-   `frontend/Dockerfile`: Multi-stage Next.js build.
