import psycopg2
from psycopg2 import extras
from datetime import datetime

class CropDiseaseDB:
    def __init__(self, dbname="crop_db", user="postgres", password="Candd65bd016.", host="localhost", port="5432"):
        self.conn_params = {
            "dbname": dbname,
            "user": user,
            "password": password,
            "host": host,
            "port": port
        }
        self.init_db()

    def get_connection(self):
        return psycopg2.connect(**self.conn_params)

    def init_db(self):
        """Initializes tables for User Information and Prediction History."""
        query_users = """
        CREATE TABLE IF NOT EXISTS users (
            user_id SERIAL PRIMARY KEY,
            username VARCHAR(50) UNIQUE NOT NULL,
            email VARCHAR(100) UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        query_history = """
        CREATE TABLE IF NOT EXISTS diagnosis_history (
            history_id SERIAL PRIMARY KEY,
            username VARCHAR(50) NOT NULL,
            image_path TEXT NOT NULL,
            predicted_disease VARCHAR(100) NOT NULL,
            confidence_score FLOAT NOT NULL,
            diagnosis_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query_users)
                cursor.execute(query_history)
                conn.commit()

    def log_diagnosis(self, username, image_path, disease, confidence):
        """Inserts a new diagnosis record into the database."""
        query = """
        INSERT INTO diagnosis_history (username, image_path, predicted_disease, confidence_score)
        VALUES (%s, %s, %s, %s);
        """
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, (username, image_path, disease, confidence))
                conn.commit()

    def get_history(self, username):
        """Retrieves prediction history for a specific user."""
        query = """
        SELECT image_path, predicted_disease, confidence_score, diagnosis_time 
        FROM diagnosis_history 
        WHERE username = %s 
        ORDER BY diagnosis_time DESC;
        """
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=extras.DictCursor) as cursor:
                cursor.execute(query, (username,))
                return cursor.fetchall()