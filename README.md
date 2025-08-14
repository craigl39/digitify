
# ✨ Digitify – Handwritten Number Recognition Web App (CRNN-based with Decimal Support)

Digitify is an end-to-end full-stack machine learning project that allows users to draw or upload handwritten numbers (e.g., 123, 45.67) and receive predictions using a custom-trained CRNN OCR model. The app includes user authentication, prediction logging, an admin dashboard, and modern CI/CD practices.

---

## 🧠 Project Stack

| Layer        | Technology        |
|--------------|-------------------|
| Frontend     | React + TypeScript + Tailwind CSS |
| Backend      | Python FastAPI    |
| ML Model     | Custom CRNN (CNN + RNN + CTC Loss) |
| Dataset      | Public handwriting datasets (IAM, SVHN, DeepOCR, or synthetic) |
| Database     | PostgreSQL        |
| Auth         | JWT-based Auth    |
| Message Queue| Celery + Redis    |
| Deployment   | Docker + GitHub Actions |
| Admin Panel  | React Dashboard for Logs |

---

## 📸 Demo

> Coming soon (insert screenshots or link to hosted app)

---

## 🚀 Features

- ✍️ Canvas or image upload for handwritten numbers (e.g., 45.67)
- 🔮 ML model predicts full string output (not just digits)
- 🧾 Logs each prediction to PostgreSQL
- 🔐 JWT user authentication system
- 📊 Admin dashboard with analytics and log viewer
- ⚙️ Async prediction with Celery + Redis
- 🐳 Dockerized for deployment
- ⚡ CI/CD with GitHub Actions

---

## 🧠 ML Model Details

### Model: CRNN (Convolutional Recurrent Neural Network)

- CNN for visual feature extraction
- Bidirectional LSTM for sequential modeling
- Fully connected + CTC loss for variable-length string output
- Predicts characters from charset: `['0'–'9', '.']`

### Dataset Sources (Recommended)

- [SVHN Dataset](http://ufldl.stanford.edu/housenumbers/)
- [IAM Handwriting Database](http://www.fki.inf.unibe.ch/databases/iam-handwriting-database)
- [DeepOCR Dataset](https://huggingface.co/datasets)
- Or generate synthetic handwriting data using fonts and PIL

---

## 🔧 Train Your Own CRNN Model

Example training script (simplified):

```python
# backend/train_crnn.py
from crnn_model import CRNN
from torch.utils.data import DataLoader
from dataset import HandwrittenNumberDataset

train_loader = DataLoader(HandwrittenNumberDataset(...), batch_size=32)
model = CRNN(num_classes=len(charset))
model.train_loop(train_loader)
model.save("app/ml_model/crnn_model.pth")
```

Inference:

```python
# app/ml_model/predict.py
def predict(image):
    # preprocess
    output = crnn_model(image)
    return decode_ctc_output(output)
```

---

## 📦 Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

---

## 🖼️ Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

---

## 🔄 Celery Worker

```bash
celery -A app.celery_worker.celery worker --loglevel=info
```

---

## 🐳 Dockerized Deployment

```bash
docker-compose up --build
```

---

## 🔐 Environment Variables

`.env.example`:

```
DATABASE_URL=postgresql://user:password@db:5432/digitify
SECRET_KEY=your_jwt_secret
REDIS_URL=redis://redis:6379/0
```

---

## 🔐 Auth & API Routes

| Method | Endpoint         | Description                  |
|--------|------------------|------------------------------|
| POST   | `/predict`       | Predict full number string   |
| POST   | `/auth/register` | Register user                |
| POST   | `/auth/login`    | Login, receive JWT           |
| GET    | `/admin/logs`    | View logs (admin only)       |

---

## 📊 Admin Dashboard

- View prediction logs
- Filter by user, digit, confidence, date
- Admin-only access

---

## ⚡ CI/CD with GitHub Actions

`.github/workflows/ci-cd.yml` handles:

- Linting
- Testing
- Docker builds

---

## ✅ Roadmap

- [x] CRNN-based full number prediction
- [ ] Scientific notation support
- [ ] Form/table field extraction
- [ ] Mobile upload
- [ ] S3 image storage
- [ ] Realtime analytics dashboard

---

## 📄 License

MIT © 2025 Craig Lofton

---

## 💬 Contact

Feel free to reach out via GitHub Issues or [craiglofton39@gmail.com].
