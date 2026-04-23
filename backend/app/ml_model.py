import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import numpy as np

# Données d'entraînement (exemples de transactions avec leurs catégories)
TRAINING_DATA = [
    # Restaurant
    ("restaurant le riad", "Restaurant"),
    ("mcdonalds", "Restaurant"),
    ("pizza hut", "Restaurant"),
    ("café starbucks", "Restaurant"),
    ("burger king", "Restaurant"),
    ("kfc poulet", "Restaurant"),
    ("sushi shop", "Restaurant"),
    ("pâtisserie", "Restaurant"),
    ("boulangerie pain", "Restaurant"),
    ("fast food", "Restaurant"),
    
    # Courses
    ("carrefour courses", "Courses"),
    ("marjane supermarché", "Courses"),
    ("épicerie du coin", "Courses"),
    ("acima shopping", "Courses"),
    ("lidl achats", "Courses"),
    ("fruits légumes marché", "Courses"),
    ("viande boucherie", "Courses"),
    ("poisson", "Courses"),
    
    # Logement
    ("loyer appartement", "Logement"),
    ("électricité lydec", "Logement"),
    ("eau redal", "Logement"),
    ("gaz butane", "Logement"),
    ("internet fibre", "Logement"),
    ("réparation plomberie", "Logement"),
    ("peinture maison", "Logement"),
    ("meubles ikea", "Logement"),
    
    # Transport
    ("essence station total", "Transport"),
    ("gasoil", "Transport"),
    ("taxi course", "Transport"),
    ("uber trajet", "Transport"),
    ("careem", "Transport"),
    ("train oncf", "Transport"),
    ("bus ctm", "Transport"),
    ("parking", "Transport"),
    ("péage autoroute", "Transport"),
    ("réparation voiture garage", "Transport"),
    
    # Abonnements
    ("netflix subscription", "Abonnements"),
    ("spotify premium", "Abonnements"),
    ("youtube premium", "Abonnements"),
    ("amazon prime", "Abonnements"),
    ("salle sport gym", "Abonnements"),
    ("abonnement téléphone", "Abonnements"),
    ("internet mensuel", "Abonnements"),
    
    # Santé
    ("pharmacie médicaments", "Santé"),
    ("médecin consultation", "Santé"),
    ("docteur clinique", "Santé"),
    ("analyses laboratoire", "Santé"),
    ("dentiste soins", "Santé"),
    ("lunettes opticien", "Santé"),
    ("assurance maladie", "Santé"),
    
    # Loisirs
    ("cinéma megarama", "Loisirs"),
    ("concert billets", "Loisirs"),
    ("voyage hotel", "Loisirs"),
    ("jeux vidéo playstation", "Loisirs"),
    ("livres librairie", "Loisirs"),
    ("sport équipement", "Loisirs"),
    
    # Vêtements
    ("zara vêtements", "Vêtements"),
    ("h&m shopping", "Vêtements"),
    ("nike chaussures", "Vêtements"),
    ("adidas", "Vêtements"),
    ("pull and bear", "Vêtements"),
    
    # Éducation
    ("université frais", "Éducation"),
    ("livres scolaires", "Éducation"),
    ("cours particuliers", "Éducation"),
    ("formation en ligne", "Éducation"),
    ("fournitures école", "Éducation"),
]

class TransactionCategorizer:
    def __init__(self):
        self.model = None
        self.model_path = "transaction_model.pkl"
        
    def train(self):
        """Entraîner le modèle ML"""
        # Séparation descriptions et catégories
        descriptions = [item[0] for item in TRAINING_DATA]
        categories = [item[1] for item in TRAINING_DATA]
        
        # Création d'un pipeline: TF-IDF + Naive Bayes
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),  # Unigrammes et bigrammes
                max_features=200
            )),
            ('classifier', MultinomialNB(alpha=0.1))
        ])
        
        
        self.model.fit(descriptions, categories)
        
        # Sauvegarder le modèle
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        print(f"Modèle entraîné et sauvegardé dans {self.model_path}")
        
    def load(self):
        """Charger le modèle sauvegardé"""
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            print("Modèle chargé")
            return True
        return False
        
    def predict(self, description: str) -> str:
        """Prédire la catégorie d'une transaction"""
        if self.model is None:
            # Essayer de charger le modèle
            if not self.load():
                # Si pas de modèle, entraîner
                self.train()
        
        # Prédire avec confiance
        prediction = self.model.predict([description.lower()])[0]
        probabilities = self.model.predict_proba([description.lower()])[0]
        confidence = max(probabilities)
        
        # Si la confiance est faible, retourner "Autres"
        if confidence < 0.3:
            return "Autres"
        
        return prediction
    
    def predict_with_confidence(self, description: str) -> tuple:
        """Prédire avec le score de confiance"""
        if self.model is None:
            if not self.load():
                self.train()
        
        prediction = self.model.predict([description.lower()])[0]
        probabilities = self.model.predict_proba([description.lower()])[0]
        confidence = max(probabilities)
        
        return prediction, confidence

# Instance globale
categorizer = TransactionCategorizer()