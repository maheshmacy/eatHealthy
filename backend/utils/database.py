import os
import json
import sqlite3
from typing import Dict, List, Any, Optional, Union, Tuple

class DataStorage:
    def __init__(self, db_path: str = "food_data.db"):
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        
    def establish_connection(self) -> None:
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        
    def close_connection(self) -> None:
        if self.conn:
            self.conn.close()
            self.conn = None
            self.cursor = None
    
    def setup_tables(self) -> None:
        self.establish_connection()
        
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            username TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS food_items (
            item_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            calories REAL,
            protein REAL,
            carbs REAL,
            fat REAL,
            user_id TEXT,
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        )
        ''')
        
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS food_logs (
            log_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            item_id TEXT NOT NULL,
            quantity REAL DEFAULT 1.0,
            meal_type TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(user_id),
            FOREIGN KEY (item_id) REFERENCES food_items(item_id)
        )
        ''')
        
        self.conn.commit()
        self.close_connection()
    
    def add_user(self, user_data: Dict[str, str]) -> bool:
        try:
            self.establish_connection()
            self.cursor.execute(
                "INSERT INTO users (user_id, username, email, password) VALUES (?, ?, ?, ?)",
                (user_data["user_id"], user_data["username"], user_data["email"], user_data["password"])
            )
            self.conn.commit()
            return True
        except sqlite3.Error:
            return False
        finally:
            self.close_connection()
    
    def authenticate_user(self, email: str, password: str) -> Optional[Dict[str, Any]]:
        self.establish_connection()
        try:
            self.cursor.execute(
                "SELECT user_id, username, email FROM users WHERE email = ? AND password = ?",
                (email, password)
            )
            result = self.cursor.fetchone()
            if result:
                return {
                    "user_id": result[0],
                    "username": result[1],
                    "email": result[2]
                }
            return None
        finally:
            self.close_connection()
    
    def add_food_item(self, food_data: Dict[str, Any]) -> bool:
        try:
            self.establish_connection()
            self.cursor.execute(
                """INSERT INTO food_items 
                   (item_id, name, category, calories, protein, carbs, fat, user_id) 
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    food_data["item_id"],
                    food_data["name"],
                    food_data["category"],
                    food_data["calories"],
                    food_data["protein"],
                    food_data["carbs"],
                    food_data["fat"],
                    food_data.get("user_id")
                )
            )
            self.conn.commit()
            return True
        except sqlite3.Error:
            return False
        finally:
            self.close_connection()
    
    def log_food_consumption(self, log_data: Dict[str, Any]) -> bool:
        try:
            self.establish_connection()
            self.cursor.execute(
                """INSERT INTO food_logs 
                   (log_id, user_id, item_id, quantity, meal_type) 
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    log_data["log_id"],
                    log_data["user_id"],
                    log_data["item_id"],
                    log_data.get("quantity", 1.0),
                    log_data.get("meal_type")
                )
            )
            self.conn.commit()
            return True
        except sqlite3.Error:
            return False
        finally:
            self.close_connection()
    
    def get_food_items(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        self.establish_connection()
        try:
            query = "SELECT * FROM food_items"
            params = []
            
            if filters:
                conditions = []
                for key, value in filters.items():
                    conditions.append(f"{key} = ?")
                    params.append(value)
                
                if conditions:
                    query += " WHERE " + " AND ".join(conditions)
            
            self.cursor.execute(query, params)
            columns = [column[0] for column in self.cursor.description]
            return [dict(zip(columns, row)) for row in self.cursor.fetchall()]
        finally:
            self.close_connection()
    
    def get_user_logs(self, user_id: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> List[Dict[str, Any]]:
        self.establish_connection()
        try:
            query = """
            SELECT l.log_id, l.user_id, l.item_id, l.quantity, l.meal_type, l.timestamp,
                   f.name, f.calories, f.protein, f.carbs, f.fat
            FROM food_logs l
            JOIN food_items f ON l.item_id = f.item_id
            WHERE l.user_id = ?
            """
            params = [user_id]
            
            if start_date:
                query += " AND l.timestamp >= ?"
                params.append(start_date)
            
            if end_date:
                query += " AND l.timestamp <= ?"
                params.append(end_date)
            
            query += " ORDER BY l.timestamp DESC"
            
            self.cursor.execute(query, params)
            columns = [column[0] for column in self.cursor.description]
            return [dict(zip(columns, row)) for row in self.cursor.fetchall()]
        finally:
            self.close_connection()
    
    def update_food_item(self, item_id: str, updated_data: Dict[str, Any]) -> bool:
        try:
            self.establish_connection()
            
            set_clause = ", ".join([f"{key} = ?" for key in updated_data.keys()])
            values = list(updated_data.values())
            values.append(item_id)
            
            query = f"UPDATE food_items SET {set_clause} WHERE item_id = ?"
            self.cursor.execute(query, values)
            self.conn.commit()
            return self.cursor.rowcount > 0
        except sqlite3.Error:
            return False
        finally:
            self.close_connection()
    
    def delete_food_log(self, log_id: str) -> bool:
        try:
            self.establish_connection()
            self.cursor.execute("DELETE FROM food_logs WHERE log_id = ?", (log_id,))
            self.conn.commit()
            return self.cursor.rowcount > 0
        except sqlite3.Error:
            return False
        finally:
            self.close_connection()

    def calculate_nutrition_summary(self, user_id: str, date: str) -> Dict[str, float]:
        logs = self.get_user_logs(user_id, date, date)
        
        summary = {
            "total_calories": 0,
            "total_protein": 0,
            "total_carbs": 0,
            "total_fat": 0
        }
        
        for log in logs:
            summary["total_calories"] += log["calories"] * log["quantity"]
            summary["total_protein"] += log["protein"] * log["quantity"]
            summary["total_carbs"] += log["carbs"] * log["quantity"]
            summary["total_fat"] += log["fat"] * log["quantity"]
        
        return summary
