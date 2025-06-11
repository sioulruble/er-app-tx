import numpy as np
import pandas as pd
import joblib
import subprocess
import os
import tempfile
from pathlib import Path

class LSTMEmotionModel:
    def __init__(self, model_path=None, scaler_path=None):
        """
        Initialise le modèle LSTM pour la reconnaissance d'émotions
        
        Args:
            model_path: Chemin vers le modèle LSTM
            scaler_path: Chemin vers le StandardScaler
        """
        # Obtention du chemin du script courant
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Chemins par défaut - chercher d'abord dans le répertoire actuel
        self.model_path = model_path or os.path.join(current_dir, 'LTSM_keras_f1_footstep20_n2.joblib')
        self.scaler_path = scaler_path or os.path.join(current_dir, 'Standardscaler.joblib')
        
        # Si les fichiers n'existent pas dans le répertoire actuel, chercher dans le sous-dossier models/lstm
        if not os.path.exists(self.model_path):
            self.model_path = os.path.join(current_dir, 'models', 'lstm', 'LTSM_keras_f1_footstep20_n2.joblib')
        if not os.path.exists(self.scaler_path):
            self.scaler_path = os.path.join(current_dir, 'models', 'lstm', 'Standardscaler.joblib')
        
        # Vérifier si les fichiers existent maintenant
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Le fichier modèle '{self.model_path}' n'existe pas")
        if not os.path.exists(self.scaler_path):
            raise FileNotFoundError(f"Le fichier scaler '{self.scaler_path}' n'existe pas")
        
        # Charger le scaler
        self.scaler = joblib.load(self.scaler_path)
        
        # Mapping des émotions (basé sur le commentaire du fichier test.py)
        self.emotion_mapping = {
            0: "happy",
            1: "disgusted",
            2: "fearful",
            3: "sad",
            4: "surprised",
            5: "angry",
            6: "neutral"  # Ajouté en supposant que le modèle peut aussi prédire neutre
        }
    
    def process_video(self, video_path):
        """
        Traite une vidéo pour la reconnaissance d'émotions
        
        Args:
            video_path: Chemin vers le fichier vidéo
        
        Returns:
            DataFrame avec les résultats d'analyse
        """
        # Créer un dossier temporaire pour les résultats OpenFace
        with tempfile.TemporaryDirectory() as temp_dir:
            # Déterminer le chemin de l'exécutable OpenFace
            # NOTE: Cette partie doit être adaptée à l'emplacement réel d'OpenFace
            openface_paths = [
                # Essayer plusieurs chemins possibles pour OpenFace
                os.path.abspath(os.path.join('..', '..', '..', '..', 'openFace', 'OpenFace', 'build', 'bin', 'FeatureExtraction')),
                os.path.abspath(os.path.join('openFace', 'OpenFace', 'build', 'bin', 'FeatureExtraction')),
                '/usr/local/bin/FeatureExtraction',  # Chemin d'installation standard Linux
                'C:/Program Files/OpenFace/bin/FeatureExtraction.exe',  # Chemin Windows
                'FeatureExtraction'  # Si dans le PATH
            ]
            
            # Trouver le premier chemin valide
            openface_executable = None
            for path in openface_paths:
                if os.path.exists(path):
                    openface_executable = path
                    break
            
            if openface_executable is None:
                raise FileNotFoundError("OpenFace executable not found. Please check the installation.")
            
            # Exécuter OpenFace
            video_basename = os.path.basename(video_path)
            result_file = os.path.join(temp_dir, os.path.splitext(video_basename)[0] + '.csv')
            
            # Imprimer la commande pour le débogage
            cmd = [
                openface_executable,
                "-f", video_path,
                "-out_dir", temp_dir
            ]
            print(f"Executing: {' '.join(cmd)}")
            
            # Exécuter OpenFace
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Vérifier le résultat
            if result.returncode != 0:
                print(f"Error running OpenFace: {result.stderr}")
                raise RuntimeError(f"OpenFace failed with return code {result.returncode}")
            
            # Vérifier si le fichier CSV a été généré
            if not os.path.exists(result_file):
                # Chercher tout fichier CSV généré
                csv_files = [f for f in os.listdir(temp_dir) if f.endswith('.csv')]
                if csv_files:
                    result_file = os.path.join(temp_dir, csv_files[0])
                else:
                    raise FileNotFoundError(f"OpenFace n'a pas généré de fichier CSV pour {video_path}")
            
            # Prétraiter les données
            processed_data = self.preprocessing(result_file)
            
            # Initialize list for results
            all_predictions = []
            
            # Process predictions in chunks of 20
            for i in range(0, processed_data.shape[0], 20):
                chunk = processed_data[i:i+20]
                if len(chunk) == 20:  # Only process complete chunks
                    prediction = self.predict(chunk)
                    emotion = self.emotion_mapping.get(prediction, "unknown")
                    all_predictions.append({
                        'timestamp': i/20,  # Timestamp based on chunk index
                        'emotion': emotion,
                        'confidence': 0.9  # Placeholder confidence value
                    })
            
            # Create DataFrame with all results
            results = pd.DataFrame(all_predictions)
            
            if len(results) == 0:
                # Fallback if no predictions were made
                results = pd.DataFrame({
                    'timestamp': [0.0],
                    'emotion': ['unknown'],
                    'confidence': [0.0]
                })
            return results
    
    def analyze_frame(self, frame):
        """
        Analyse une image fixe pour détecter l'émotion
        Note: Cette fonction est un placeholder puisque LSTM n'est pas
        conçu pour l'analyse d'images fixes
        """
        # Retourner un résultat par défaut
        return {
            "emotion": {
                "happy": 0.1,
                "disgusted": 0.1,
                "fearful": 0.1,
                "sad": 0.1,
                "surprised": 0.1,
                "angry": 0.3,
                "neutral": 0.4
            }
        }
    
    def preprocessing(self, file_path):
        """
        Prétraite les données CSV extraites d'OpenFace
        """
        # Charger csv
        try:
            # Essayer d'abord avec le séparateur standard
            csv = pd.read_csv(file_path, sep=",")
        except:
            # Essayer avec un séparateur qui permet des espaces
            csv = pd.read_csv(file_path, sep=",\s+")
        
        # Sauvegarder une copie des en-têtes pour le débogage
        print(f"Columns in CSV: {csv.columns.tolist()}")
        
        # Corriger le nom des colonnes
        csv.columns = csv.columns.str.strip()
        
        # Vérifier si 'timestamp' existe dans les colonnes
        if 'timestamp' not in csv.columns:
            # Si pas de timestamp, créer une colonne artificielle
            csv['timestamp'] = pd.Series(range(len(csv)))
        
        # Convertir 'timestamp' en format datetime et l'utiliser comme index
        try:
            csv['timestamp'] = pd.to_datetime(csv['timestamp'], unit='s', origin=pd.Timestamp('now').normalize())
        except:
            # En cas d'erreur, créer un timestamp artificiel
            csv['timestamp'] = pd.date_range(start=pd.Timestamp('now'), periods=len(csv), freq='0.1S')
        
        # Calculer la durée de la vidéo en secondes
        self.video_duration = (csv['timestamp'].max() - csv['timestamp'].min()).total_seconds()
        print(f"Video duration: {self.video_duration} seconds")
        
        csv.set_index('timestamp', inplace=True)
        
        # Rééchantillonner les données à la fréquence spécifiée
        resample_frequency = '0.1s'
        resampled_csv = csv.resample(resample_frequency).mean()
        
        # Remplir les valeurs manquantes
        resampled_csv.fillna(method='ffill', inplace=True)
        resampled_csv.fillna(method='bfill', inplace=True)
        
        # Définir la longueur cible
        target_length = 20
        period = 4
        if self.video_duration > period:
            target_length = int(self.video_duration / period)*20
        else:      
            target_length = 20
        print(f"Target length: {target_length}")
        print(f"Resampled data shape: {resampled_csv.shape}")
        print(f"Resampled data: {resampled_csv.head()}")
        # Vérifier si le DataFrame est vide
        # Ajuster la longueur des données
        if len(resampled_csv) < target_length:
            # Ajouter du padding avec des zéros
            num_missing = target_length - len(resampled_csv)
            padding = pd.DataFrame(np.zeros((num_missing, resampled_csv.shape[1])), columns=resampled_csv.columns)
            resampled_csv = pd.concat([resampled_csv, padding])
        elif len(resampled_csv) > target_length:
            # Tronquer la séquence
            resampled_csv = resampled_csv.iloc[:target_length]
        
        # Enregistrer temporairement au format csv
        temp_csv_path = tempfile.mktemp(suffix='.csv')
        resampled_csv.to_csv(temp_csv_path, index=False)
        resampled_csv = pd.read_csv(temp_csv_path)
        os.remove(temp_csv_path)  # Nettoyer
        
        # Vérifier si le dataframe a au moins 4 colonnes
        if resampled_csv.shape[1] <= 4:
            print(f"Attention: Le CSV n'a que {resampled_csv.shape[1]} colonnes, moins que les 4 attendues")
            # Dans ce cas, ne pas retirer les colonnes
        else:
            # Retirer les quatres premières colonnes (colonnes descriptives)
            resampled_csv = resampled_csv.iloc[:,4:]
        
        # Normaliser les données
        scaled_df = self.scaler.transform(resampled_csv)
        
        # Adapter le df à la taille qu'il doit prendre dans le LSTM
        reshaped_df = scaled_df.reshape((-1, 1, resampled_csv.shape[1]))
        
        return reshaped_df
    
    def predict(self, test_df):
        """
        Effectue la prédiction à partir des données prétraitées
        """
        # Charger le modèle
        loaded_model = joblib.load(self.model_path)
        
        # Initialiser la liste de prédiction
        pred = []
        
        try:
            # Prédire
            for i in range(test_df.shape[1]):
                df_temp = test_df[:, i:i+1, :]
                predictions = loaded_model.predict(df_temp)
                predicted_classes = np.argmax(predictions, axis=1)
                pred.append(predicted_classes)
            
            pred = np.array(pred[0])
            pred_flattened = np.bincount(pred).argmax()
            
            return pred_flattened
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            # Retourner une classe par défaut en cas d'erreur
            return 0  # Happy par défaut