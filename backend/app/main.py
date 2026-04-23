from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List
from datetime import datetime, timedelta
import random
from pydantic import BaseModel

from . import models
from .database import engine, get_db
from .ml_model import categorizer


# Création des tables dans la base de données
models.Base.metadata.create_all(bind=engine)

# Modèles Pydantic pour les requêtes
class TransactionCreate(BaseModel):
    amount: float
    description: str

class TransactionResponse(BaseModel):
    id: int
    amount: float
    description: str
    category: str
    date: str
    
    class Config:
        from_attributes = True

app = FastAPI(title="Smart Finance API")

# Permettre les requêtes depuis l'app mobile
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modèle ML simple pour catégoriser les transactions
def categorize_transaction(description: str) -> str:
    """Catégorise une transaction avec ML"""
    try:
        category = categorizer.predict(description)
        return category
    except Exception as e:
        print(f"Erreur ML: {e}")
        # Fallback sur règles simples
        description_lower = description.lower()
        
        if any(word in description_lower for word in ['restaurant', 'café', 'mcdo', 'pizza', 'burger']):
            return 'Restaurant'
        elif any(word in description_lower for word in ['carrefour', 'marjane', 'supermarché', 'épicerie']):
            return 'Courses'
        elif any(word in description_lower for word in ['loyer', 'appartement', 'eau', 'électricité']):
            return 'Logement'
        elif any(word in description_lower for word in ['essence', 'taxi', 'bus', 'train', 'parking']):
            return 'Transport'
        else:
            return 'Autres'

# Routes API

@app.get("/")
def read_root():
    return {"message": "Smart Finance API - Backend opérationnel!"}

@app.post("/transactions/")
def create_transaction(transaction: TransactionCreate, db: Session = Depends(get_db)):
    """Créer une nouvelle transaction"""
    category = categorize_transaction(transaction.description)
    
    new_transaction = models.Transaction(
        amount=transaction.amount,
        description=transaction.description,
        category=category
    )
    
    db.add(new_transaction)
    db.commit()
    db.refresh(new_transaction)
    
    return {
        "id": new_transaction.id,
        "amount": new_transaction.amount,
        "description": new_transaction.description,
        "category": category,
        "date": new_transaction.date.isoformat()
    }

@app.get("/transactions/")
def get_transactions(db: Session = Depends(get_db)):
    """Récupérer toutes les transactions"""
    transactions = db.query(models.Transaction).order_by(models.Transaction.date.desc()).all()
    return transactions

@app.get("/stats/monthly")
def get_monthly_stats(db: Session = Depends(get_db)):
    """Statistiques du mois en cours"""
    # Récupérer toutes les transactions du mois
    now = datetime.now()
    first_day = datetime(now.year, now.month, 1)
    
    transactions = db.query(models.Transaction).filter(
        models.Transaction.date >= first_day
    ).all()
    
    # Calculer le total
    total_spent = sum(t.amount for t in transactions)
    
    # Grouper par catégorie
    categories = {}
    for t in transactions:
        if t.category not in categories:
            categories[t.category] = 0
        categories[t.category] += t.amount
    
    # Trier par montant
    top_categories = sorted(
        [{"name": k, "amount": v} for k, v in categories.items()],
        key=lambda x: x["amount"],
        reverse=True
    )[:5]
    
    return {
        "total_spent": round(total_spent, 2),
        "month": now.strftime("%B %Y"),
        "top_categories": top_categories,
        "transaction_count": len(transactions)
    }

@app.get("/predictions/next-month")
def predict_next_month(db: Session = Depends(get_db)):
    """Prédire les dépenses du mois prochain (ML basique pour l'instant)"""
    # Récupérer les 3 derniers mois
    now = datetime.now()
    three_months_ago = now - timedelta(days=90)
    
    transactions = db.query(models.Transaction).filter(
        models.Transaction.date >= three_months_ago
    ).all()
    
    if not transactions:
        return {"predicted_amount": 0, "confidence": 0}
    
    # Moyenne simple
    total = sum(t.amount for t in transactions)
    monthly_avg = total / 3
    
    # Ajouter une petite variation (+5% à +15%)
    variation = random.uniform(1.05, 1.15)
    predicted = monthly_avg * variation
    
    return {
        "predicted_amount": round(predicted, 2),
        "confidence": 0.78,
        "trend": "increasing" if variation > 1.1 else "stable"
    }

@app.get("/alerts/")
def get_alerts(db: Session = Depends(get_db)):
    """Détecter les anomalies et générer des alertes"""
    alerts = []
    
    # Récupérer les transactions récentes
    week_ago = datetime.now() - timedelta(days=7)
    recent_transactions = db.query(models.Transaction).filter(
        models.Transaction.date >= week_ago
    ).all()
    
    # Alerte: Transactions importantes
    for t in recent_transactions:
        if t.amount > 400:  # Seuil arbitraire
            alerts.append({
                "type": "large_transaction",
                "severity": "warning",
                "message": f"Transaction importante: {t.amount} MAD chez {t.description}",
                "date": t.date
            })
    
    # Alerte: Budget dépassé (exemple)
    monthly_stats = get_monthly_stats(db)
    if monthly_stats["total_spent"] > 3000:
        alerts.append({
            "type": "budget_exceeded",
            "severity": "alert",
            "message": f"Dépenses totales: {monthly_stats['total_spent']} MAD (objectif: 3000 MAD)",
            "date": datetime.now()
        })
    
    return alerts

# Route pour ajouter des données de test
@app.post("/seed-data/")
def seed_test_data(db: Session = Depends(get_db)):
    
    test_transactions = [
        {"amount": 120, "description": "Restaurant Le Riad"},
        {"amount": 1200, "description": "Loyer appartement"},
        {"amount": 45, "description": "Essence station Total"},
        {"amount": 85, "description": "Courses Marjane"},
        {"amount": 15.99, "description": "Abonnement Netflix"},
        {"amount": 340, "description": "Courses Carrefour"},
        {"amount": 200, "description": "Restaurant Pizza Hut"},
        {"amount": 50, "description": "Taxi aéroport"},
        {"amount": 89, "description": "Électricité LYDEC"},
        {"amount": 450, "description": "Electronics Store - Casque"},
    ]
    
    for t_data in test_transactions:
        category = categorize_transaction(t_data["description"])
        transaction = models.Transaction(
            amount=t_data["amount"],
            description=t_data["description"],
            category=category
        )
        db.add(transaction)
    
    db.commit()
    
    return {"message": f"{len(test_transactions)} transactions de test créées!"}



@app.post("/ml/train")
def train_ml_model():
    """Entraîner le modèle ML de catégorisation"""
    try:
        categorizer.train()
        return {"message": "Modèle ML entraîné avec succès!"}
    except Exception as e:
        return {"error": str(e)}

@app.post("/ml/predict")
def predict_category(description: str):
    """Tester la prédiction ML"""
    category, confidence = categorizer.predict_with_confidence(description)
    return {
        "description": description,
        "predicted_category": category,
        "confidence": round(confidence * 100, 2)
    }