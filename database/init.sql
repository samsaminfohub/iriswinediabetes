-- Création de la base de données (si elle n'existe pas)
CREATE DATABASE IF NOT EXISTS mlflow_predictions;

-- Connexion à la base de données
\c mlflow_predictions;

-- Création de la table principale pour les prédictions
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    dataset_type VARCHAR(50) NOT NULL CHECK (dataset_type IN ('iris', 'wine', 'diabetes')),
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50),
    input_features JSONB NOT NULL,
    prediction_result JSONB NOT NULL,
    confidence_score FLOAT CHECK (confidence_score >= 0 AND confidence_score <= 1),
    prediction_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    user_id VARCHAR(100),
    experiment_id VARCHAR(100),
    run_id VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index pour améliorer les performances des requêtes
CREATE INDEX IF NOT EXISTS idx_predictions_dataset_type ON predictions(dataset_type);
CREATE INDEX IF NOT EXISTS idx_predictions_model_name ON predictions(model_name);
CREATE INDEX IF NOT EXISTS idx_predictions_time ON predictions(prediction_time);
CREATE INDEX IF NOT EXISTS idx_predictions_experiment_id ON predictions(experiment_id);
CREATE INDEX IF NOT EXISTS idx_predictions_user_id ON predictions(user_id);

-- Index composite pour les requêtes fréquentes
CREATE INDEX IF NOT EXISTS idx_predictions_dataset_model ON predictions(dataset_type, model_name);
CREATE INDEX IF NOT EXISTS idx_predictions_time_desc ON predictions(prediction_time DESC);

-- Table pour stocker les métadonnées des modèles
CREATE TABLE IF NOT EXISTS model_metadata (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL UNIQUE,
    dataset_type VARCHAR(50) NOT NULL,
    model_type VARCHAR(50), -- 'classification', 'regression'
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    total_predictions INTEGER DEFAULT 0,
    average_confidence FLOAT,
    is_active BOOLEAN DEFAULT TRUE
);

-- Table pour les statistiques de performance
CREATE TABLE IF NOT EXISTS performance_metrics (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    dataset_type VARCHAR(50) NOT NULL,
    metric_name VARCHAR(50) NOT NULL, -- 'accuracy', 'precision', 'recall', 'f1', 'mse', 'mae'
    metric_value FLOAT NOT NULL,
    evaluation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    test_size INTEGER,
    notes TEXT,
    FOREIGN KEY (model_name, dataset_type) REFERENCES model_metadata(model_name, dataset_type)
);

-- Vue pour les statistiques rapides
CREATE OR REPLACE VIEW prediction_stats AS
SELECT 
    dataset_type,
    model_name,
    COUNT(*) as total_predictions,
    AVG(confidence_score) as avg_confidence,
    MIN(prediction_time) as first_prediction,
    MAX(prediction_time) as last_prediction,
    COUNT(DISTINCT user_id) as unique_users,
    COUNT(DISTINCT DATE(prediction_time)) as active_days
FROM predictions 
GROUP BY dataset_type, model_name
ORDER BY total_predictions DESC;

-- Vue pour les prédictions récentes
CREATE OR REPLACE VIEW recent_predictions AS
SELECT 
    id,
    dataset_type,
    model_name,
    prediction_result->>'predicted_class' as predicted_class,
    prediction_result->>'predicted_value' as predicted_value,
    confidence_score,
    prediction_time,
    user_id
FROM predictions 
WHERE prediction_time >= NOW() - INTERVAL '7 days'
ORDER BY prediction_time DESC;

-- Fonction pour nettoyer les anciennes prédictions
CREATE OR REPLACE FUNCTION cleanup_old_predictions(days_to_keep INTEGER DEFAULT 30)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM predictions 
    WHERE prediction_time < NOW() - INTERVAL '1 day' * days_to_keep;
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    
    -- Mettre à jour les statistiques des modèles
    UPDATE model_metadata 
    SET total_predictions = (
        SELECT COUNT(*) 
        FROM predictions 
        WHERE predictions.model_name = model_metadata.model_name
    ),
    average_confidence = (
        SELECT AVG(confidence_score) 
        FROM predictions 
        WHERE predictions.model_name = model_metadata.model_name
          AND confidence_score IS NOT NULL
    );
    
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Trigger pour mettre à jour les métadonnées des modèles
CREATE OR REPLACE FUNCTION update_model_metadata()
RETURNS TRIGGER AS $$
BEGIN
    -- Insérer ou mettre à jour les métadonnées du modèle
    INSERT INTO model_metadata (model_name, dataset_type, last_used, total_predictions)
    VALUES (NEW.model_name, NEW.dataset_type, NEW.prediction_time, 1)
    ON CONFLICT (model_name) 
    DO UPDATE SET
        last_used = NEW.prediction_time,
        total_predictions = model_metadata.total_predictions + 1,
        average_confidence = (
            SELECT AVG(confidence_score) 
            FROM predictions 
            WHERE model_name = NEW.model_name 
              AND confidence_score IS NOT NULL
        );
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Créer le trigger
DROP TRIGGER IF EXISTS trigger_update_model_metadata ON predictions;
CREATE TRIGGER trigger_update_model_metadata
    AFTER INSERT ON predictions
    FOR EACH ROW
    EXECUTE FUNCTION update_model_metadata();

-- Insérer quelques données d'exemple (optionnel)
INSERT INTO model_metadata (model_name, dataset_type, model_type, description) VALUES
('RandomForest_Iris', 'iris', 'classification', 'Random Forest classifier for Iris dataset'),
('SVM_Wine', 'wine', 'classification', 'Support Vector Machine for Wine classification'),
('LinearRegression_Diabetes', 'diabetes', 'regression', 'Linear Regression for Diabetes prediction'),
('XGBoost_Iris', 'iris', 'classification', 'XGBoost classifier for Iris dataset'),
('NeuralNetwork_Wine', 'wine', 'classification', 'Neural Network for Wine classification')
ON CONFLICT (model_name) DO NOTHING;

-- Fonction pour obtenir les statistiques d'un modèle
CREATE OR REPLACE FUNCTION get_model_stats(model_name_param VARCHAR)
RETURNS TABLE(
    total_predictions BIGINT,
    avg_confidence NUMERIC,
    first_prediction TIMESTAMP,
    last_prediction TIMESTAMP,
    daily_average NUMERIC,
    most_common_class TEXT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        COUNT(*) as total_predictions,
        ROUND(AVG(p.confidence_score), 3) as avg_confidence,
        MIN(p.prediction_time) as first_prediction,
        MAX(p.prediction_time) as last_prediction,
        ROUND(
            COUNT(*)::NUMERIC / 
            GREATEST(EXTRACT(days FROM (MAX(p.prediction_time) - MIN(p.prediction_time))), 1),
            2
        ) as daily_average,
        MODE() WITHIN GROUP (ORDER BY p.prediction_result->>'predicted_class') as most_common_class
    FROM predictions p
    WHERE p.model_name = model_name_param;
END;
$$ LANGUAGE plpgsql;

-- Permissions (ajustez selon vos besoins)
-- GRANT ALL PRIVILEGES ON DATABASE mlflow_predictions TO your_app_user;
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO your_app_user;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO your_app_user;

COMMENT ON TABLE predictions IS 'Table principale pour stocker toutes les prédictions ML';
COMMENT ON TABLE model_metadata IS 'Métadonnées et statistiques des modèles ML';
COMMENT ON TABLE performance_metrics IS 'Métriques de performance des modèles';
COMMENT ON VIEW prediction_stats IS 'Vue résumée des statistiques par modèle';
COMMENT ON VIEW recent_predictions IS 'Vue des prédictions des 7 derniers jours';