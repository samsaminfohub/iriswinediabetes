#!/usr/bin/env python3
"""
Utilitaires pour la gestion et l'analyse de la base de données ML
Usage: python database_utils.py [command]
"""

import mysql.connector
import pandas as pd
import argparse
import os
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import seaborn as sns

# Charger les variables d'environnement
load_dotenv()

class DatabaseUtils:
    """Classe utilitaire pour la gestion de la base de données"""
    
    def __init__(self):
        self.config = {
            'host': os.getenv("DB_HOST", "localhost"),
            'user': os.getenv("DB_USER", "root"),
            'password': os.getenv("DB_PASSWORD", "root_password"),
            'database': os.getenv("DB_NAME", "mlflow_results")
        }
    
    def get_connection(self):
        """Obtenir une connexion à la base de données"""
        return mysql.connector.connect(**self.config)
    
    def execute_query(self, query, params=None):
        """Exécuter une requête et retourner les résultats"""
        try:
            conn = self.get_connection()
            df = pd.read_sql(query, conn, params=params)
            conn.close()
            return df
        except Exception as e:
            print(f"❌ Erreur lors de l'exécution de la requête: {e}")
            return None
    
    def test_connection(self):
        """Tester la connexion à la base de données"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            cursor.close()
            conn.close()
            
            if result:
                print("✅ Connexion à la base de données réussie")
                return True
            else:
                print("❌ Échec de la connexion à la base de données")
                return False
        except Exception as e:
            print(f"❌ Erreur de connexion: {e}")
            return False
    
    def get_database_status(self):
        """Obtenir le statut de la base de données"""
        query = """
        SELECT 
            table_name,
            table_rows,
            ROUND(((data_length + index_length) / 1024 / 1024), 2) AS 'taille_mb'
        FROM information_schema.tables 
        WHERE table_schema = %s
        ORDER BY table_rows DESC
        """
        
        df = self.execute_query(query, (self.config['database'],))
        if df is not None:
            print("\n📊 Statut de la base de données:")
            print(df.to_string(index=False))
        return df
    
    def get_performance_summary(self):
        """Résumé des performances des modèles"""
        query = """
        SELECT 
            dataset_name,
            model_name,
            COUNT(*) as nb_executions,
            ROUND(AVG(accuracy), 4) as accuracy_moyenne,
            ROUND(MAX(accuracy), 4) as meilleure_accuracy,
            ROUND(MIN(accuracy), 4) as accuracy_min,
            ROUND(STDDEV(accuracy), 4) as ecart_type
        FROM resultats_ml
        GROUP BY dataset_name, model_name
        ORDER BY dataset_name, accuracy_moyenne DESC
        """
        
        df = self.execute_query(query)
        if df is not None:
            print("\n🎯 Résumé des performances:")
            print(df.to_string(index=False))
        return df
    
    def get_recent_activity(self, days=7):
        """Activité récente dans la base de données"""
        query = """
        SELECT 
            DATE(date_creation) as jour,
            dataset_name,
            model_name,
            COUNT(*) as nb_executions,
            ROUND(AVG(accuracy), 4) as accuracy_moyenne
        FROM resultats_ml
        WHERE date_creation >= DATE_SUB(NOW(), INTERVAL %s DAY)
        GROUP BY DATE(date_creation), dataset_name, model_name
        ORDER BY jour DESC, dataset_name, model_name
        """
        
        df = self.execute_query(query, (days,))
        if df is not None:
            print(f"\n📅 Activité des {days} derniers jours:")
            print(df.to_string(index=False))
        return df
    
    def compare_models(self, dataset_name=None):
        """Comparaison détaillée des modèles"""
        if dataset_name:
            query = """
            SELECT 
                model_name,
                COUNT(*) as nb_executions,
                ROUND(AVG(accuracy), 4) as accuracy_moyenne,
                ROUND(MAX(accuracy), 4) as meilleure_accuracy,
                ROUND(STDDEV(accuracy), 4) as ecart_type,
                ROUND(STDDEV(accuracy)/AVG(accuracy) * 100, 2) as coefficient_variation
            FROM resultats_ml
            WHERE dataset_name = %s
            GROUP BY model_name
            ORDER BY accuracy_moyenne DESC
            """
            params = (dataset_name,)
        else:
            query = """
            SELECT 
                dataset_name,
                model_name,
                COUNT(*) as nb_executions,
                ROUND(AVG(accuracy), 4) as accuracy_moyenne,
                ROUND(MAX(accuracy), 4) as meilleure_accuracy,
                ROUND(STDDEV(accuracy), 4) as ecart_type
            FROM resultats_ml
            GROUP BY dataset_name, model_name
            ORDER BY dataset_name, accuracy_moyenne DESC
            """
            params = None
        
        df = self.execute_query(query, params)
        if df is not None:
            if dataset_name:
                print(f"\n🔍 Comparaison des modèles pour {dataset_name}:")
            else:
                print("\n🔍 Comparaison globale des modèles:")
            print(df.to_string(index=False))
        return df
    
    def plot_performance_trends(self, dataset_name=None, save_path=None):
        """Graphique des tendances de performance"""
        if dataset_name:
            query = """
            SELECT 
                date_creation,
                model_name,
                accuracy
            FROM resultats_ml
            WHERE dataset_name = %s
            ORDER BY date_creation
            """
            params = (dataset_name,)
            title = f"Évolution des performances - {dataset_name}"
        else:
            query = """
            SELECT 
                date_creation,
                CONCAT(dataset_name, ' - ', model_name) as model_name,
                accuracy
            FROM resultats_ml
            ORDER BY date_creation
            """
            params = None
            title = "Évolution des performances - Tous les datasets"
        
        df = self.execute_query(query, params)
        
        if df is not None and len(df) > 0:
            plt.figure(figsize=(12, 8))
            
            for model in df['model_name'].unique():
                model_data = df[df['model_name'] == model]
                plt.plot(model_data['date_creation'], model_data['accuracy'], 
                        marker='o', label=model, linewidth=2, markersize=6)
            
            plt.title(title, fontsize=16, fontweight='bold')
            plt.xlabel('Date', fontsize=12)
            plt.ylabel('Accuracy', fontsize=12)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"📊 Graphique sauvegardé: {save_path}")
            else:
                plt.show()
        else:
            print("❌ Aucune donnée trouvée pour le graphique")
    
    def export_results(self, format='csv', output_dir='exports'):
        """Exporter les résultats dans différents formats"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Exporter les statistiques
        stats_df = self.execute_query("SELECT * FROM statistiques ORDER BY dataset_name, colonne")
        if stats_df is not None:
            if format == 'csv':
                file_path = f"{output_dir}/statistiques_{timestamp}.csv"
                stats_df.to_csv(file_path, index=False)
            elif format == 'json':
                file_path = f"{output_dir}/statistiques_{timestamp}.json"
                stats_df.to_json(file_path, orient='records', indent=2)
            elif format == 'excel':
                file_path = f"{output_dir}/statistiques_{timestamp}.xlsx"
                stats_df.to_excel(file_path, index=False)
            
            print(f"📤 Statistiques exportées: {file_path}")
        
        # Exporter les résultats ML
        ml_df = self.execute_query("SELECT * FROM resultats_ml ORDER BY dataset_name, model_name, date_creation")
        if ml_df is not None:
            if format == 'csv':
                file_path = f"{output_dir}/resultats_ml_{timestamp}.csv"
                ml_df.to_csv(file_path, index=False)
            elif format == 'json':
                file_path = f"{output_dir}/resultats_ml_{timestamp}.json"
                ml_df.to_json(file_path, orient='records', indent=2)
            elif format == 'excel':
                file_path = f"{output_dir}/resultats_ml_{timestamp}.xlsx"
                ml_df.to_excel(file_path, index=False)
            
            print(f"📤 Résultats ML exportés: {file_path}")
        
        # Exporter les expériences MLflow
        mlflow_df = self.execute_query("SELECT * FROM experiences_mlflow ORDER BY experiment_name, date_creation")
        if mlflow_df is not None:
            if format == 'csv':
                file_path = f"{output_dir}/experiences_mlflow_{timestamp}.csv"
                mlflow_df.to_csv(file_path, index=False)
            elif format == 'json':
                file_path = f"{output_dir}/experiences_mlflow_{timestamp}.json"
                mlflow_df.to_json(file_path, orient='records', indent=2)
            elif format == 'excel':
                file_path = f"{output_dir}/experiences_mlflow_{timestamp}.xlsx"
                mlflow_df.to_excel(file_path, index=False)
            
            print(f"📤 Expériences MLflow exportées: {file_path}")
    
    def cleanup_old_data(self, days=30, dry_run=True):
        """Nettoyer les anciennes données"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        tables = ['statistiques', 'resultats_ml', 'experiences_mlflow']
        
        for table in tables:
            query = f"""
            SELECT COUNT(*) as count
            FROM {table}
            WHERE date_creation < %s
            """
            
            df = self.execute_query(query, (cutoff_date,))
            if df is not None and len(df) > 0:
                count = df.iloc[0]['count']
                
                if dry_run:
                    print(f"🗑️  [DRY RUN] {count} enregistrements seraient supprimés de {table}")
                else:
                    delete_query = f"""
                    DELETE FROM {table}
                    WHERE date_creation < %s
                    """
                    
                    try:
                        conn = self.get_connection()
                        cursor = conn.cursor()
                        cursor.execute(delete_query, (cutoff_date,))
                        conn.commit()
                        cursor.close()
                        conn.close()
                        print(f"🗑️  {count} enregistrements supprimés de {table}")
                    except Exception as e:
                        print(f"❌ Erreur lors de la suppression dans {table}: {e}")
    
    def generate_report(self, output_file="rapport_ml.html"):
        """Générer un rapport HTML complet"""
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Rapport d'Analyse ML</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1, h2 { color: #333; }
                table { border-collapse: collapse; width: 100%; margin: 20px 0; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                .metric { background-color: #e8f4fd; padding: 10px; margin: 10px 0; border-radius: 5px; }
            </style>
        </head>
        <body>
            <h1>📊 Rapport d'Analyse Machine Learning</h1>
            <p>Généré le: {timestamp}</p>
        """.format(timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        # Statut de la base de données
        status_df = self.get_database_status()
        if status_df is not None:
            html_content += "<h2>📈 Statut de la Base de Données</h2>"
            html_content += status_df.to_html(index=False, classes="table")
        
        # Résumé des performances
        perf_df = self.get_performance_summary()
        if perf_df is not None:
            html_content += "<h2>🎯 Résumé des Performances</h2>"
            html_content += perf_df.to_html(index=False, classes="table")
        
        # Activité récente
        activity_df = self.get_recent_activity()
        if activity_df is not None:
            html_content += "<h2>📅 Activité Récente (7 derniers jours)</h2>"
            html_content += activity_df.to_html(index=False, classes="table")
        
        html_content += """
        </body>
        </html>
        """
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"📄 Rapport généré: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Utilitaires de base de données ML")
    parser.add_argument('command', choices=[
        'test', 'status', 'summary', 'activity', 'compare', 
        'plot', 'export', 'cleanup', 'report'
    ], help="Commande à exécuter")
    
    parser.add_argument('--dataset', type=str, help="Nom du dataset (pour certaines commandes)")
    parser.add_argument('--days', type=int, default=7, help="Nombre de jours (pour activity/cleanup)")
    parser.add_argument('--format', choices=['csv', 'json', 'excel'], default='csv', help="Format d'export")
    parser.add_argument('--output', type=str, help="Fichier de sortie")
    parser.add_argument('--dry-run', action='store_true', help="Simulation (pour cleanup)")
    
    args = parser.parse_args()
    
    db_utils = DatabaseUtils()
    
    if args.command == 'test':
        db_utils.test_connection()
    
    elif args.command == 'status':
        db_utils.get_database_status()
    
    elif args.command == 'summary':
        db_utils.get_performance_summary()
    
    elif args.command == 'activity':
        db_utils.get_recent_activity(args.days)
    
    elif args.command == 'compare':
        db_utils.compare_models(args.dataset)
    
    elif args.command == 'plot':
        db_utils.plot_performance_trends(args.dataset, args.output)
    
    elif args.command == 'export':
        db_utils.export_results(args.format)
    
    elif args.command == 'cleanup':
        db_utils.cleanup_old_data(args.days, args.dry_run)
    
    elif args.command == 'report':
        output_file = args.output or "rapport_ml.html"
        db_utils.generate_report(output_file)

if __name__ == "__main__":
    main()