-- Créer la base de données si elle n'existe pas
CREATE DATABASE IF NOT EXISTS mlflow_results;
USE mlflow_results;

-- Créer les tables pour les résultats ML
CREATE TABLE IF NOT EXISTS statistiques (
    id INT AUTO_INCREMENT PRIMARY KEY,
    dataset_name VARCHAR(100) NOT NULL,
    colonne VARCHAR(100) NOT NULL,
    moyenne FLOAT,
    ecart_type FLOAT,
    minimum FLOAT,
    maximum FLOAT,
    mediane FLOAT,
    q1 FLOAT,
    q3 FLOAT,
    date_creation TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_dataset_colonne (dataset_name, colonne),
    INDEX idx_date_creation (date_creation)
);

CREATE TABLE IF NOT EXISTS resultats_ml (
    id INT AUTO_INCREMENT PRIMARY KEY,
    dataset_name VARCHAR(100) NOT NULL,
    model_name VARCHAR(100) NOT NULL,
    accuracy FLOAT,
    precision_macro FLOAT,
    recall_macro FLOAT,
    f1_score_macro FLOAT,
    confusion_matrix TEXT,
    hyperparameters TEXT,
    date_creation TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_dataset_model (dataset_name, model_name),
    INDEX idx_accuracy (accuracy),
    INDEX idx_date_creation (date_creation)
);

CREATE TABLE IF NOT EXISTS experiences_mlflow (
    id INT AUTO_INCREMENT PRIMARY KEY,
    run_id VARCHAR(255) UNIQUE NOT NULL,
    experiment_name VARCHAR(100) NOT NULL,
    dataset_name VARCHAR(100) NOT NULL,
    model_name VARCHAR(100) NOT NULL,
    accuracy FLOAT,
    parameters TEXT,
    metrics TEXT,
    date_creation TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_run_id (run_id),
    INDEX idx_experiment (experiment_name),
    INDEX idx_dataset_model (dataset_name, model_name),
    INDEX idx_date_creation (date_creation)
);

-- Insérer quelques données d'exemple
INSERT INTO statistiques (dataset_name, colonne, moyenne, ecart_type, minimum, maximum, mediane, q1, q3) VALUES
('Iris', 'sepal_length', 5.84, 0.83, 4.3, 7.9, 5.8, 5.1, 6.4),
('Iris', 'sepal_width', 3.06, 0.44, 2.0, 4.4, 3.0, 2.8, 3.3),
('Wine', 'alcohol', 13.0, 0.81, 11.03, 14.83, 13.05, 12.36, 13.68),
('Diabetes', 'glucose', 120.89, 31.97, 0.0, 199.0, 117.0, 99.0, 140.25);