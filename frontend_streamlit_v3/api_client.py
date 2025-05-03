import requests
import json
from typing import Dict, Any, List, Optional, Union

class NutritionAPI:
    def __init__(self, api_endpoint: str, api_key: Optional[str] = None):
        self.endpoint = api_endpoint
        self.key = api_key
        self.session = requests.Session()
        if self.key:
            self.session.headers.update({"Authorization": f"Bearer {self.key}"})
    
    def fetch_nutritional_data(self, food_query: str) -> Dict[str, Any]:
        try:
            response = self.session.get(
                f"{self.endpoint}/nutrition",
                params={"query": food_query}
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e), "status": "failed"}
    
    def search_food_items(self, search_term: str, limit: int = 10) -> List[Dict[str, Any]]:
        try:
            response = self.session.get(
                f"{self.endpoint}/search",
                params={"term": search_term, "limit": limit}
            )
            response.raise_for_status()
            return response.json().get("items", [])
        except requests.exceptions.RequestException as e:
            return [{"error": str(e), "status": "failed"}]
    
    def get_food_details(self, food_id: str) -> Dict[str, Any]:
        try:
            response = self.session.get(
                f"{self.endpoint}/food/{food_id}"
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e), "status": "failed"}
    
    def analyze_meal(self, meal_components: List[Dict[str, Union[str, float]]]) -> Dict[str, Any]:
        try:
            response = self.session.post(
                f"{self.endpoint}/analyze",
                json={"components": meal_components}
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e), "status": "failed"}
    
    def get_recommendations(self, user_profile: Dict[str, Any]) -> Dict[str, Any]:
        try:
            response = self.session.post(
                f"{self.endpoint}/recommend",
                json=user_profile
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e), "status": "failed"}

class UserData:
    def __init__(self, data_service_url: str, user_token: Optional[str] = None):
        self.service_url = data_service_url
        self.token = user_token
        self.session = requests.Session()
        if self.token:
            self.session.headers.update({"Authorization": f"Bearer {self.token}"})
    
    def authenticate(self, username: str, password: str) -> Dict[str, Any]:
        try:
            response = self.session.post(
                f"{self.service_url}/auth",
                json={"username": username, "password": password}
            )
            response.raise_for_status()
            auth_data = response.json()
            if "token" in auth_data:
                self.token = auth_data["token"]
                self.session.headers.update({"Authorization": f"Bearer {self.token}"})
            return auth_data
        except requests.exceptions.RequestException as e:
            return {"error": str(e), "status": "failed"}
    
    def get_user_profile(self) -> Dict[str, Any]:
        if not self.token:
            return {"error": "Not authenticated", "status": "failed"}
        
        try:
            response = self.session.get(f"{self.service_url}/profile")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e), "status": "failed"}
    
    def update_user_profile(self, profile_data: Dict[str, Any]) -> Dict[str, Any]:
        if not self.token:
            return {"error": "Not authenticated", "status": "failed"}
        
        try:
            response = self.session.put(
                f"{self.service_url}/profile",
                json=profile_data
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e), "status": "failed"}
    
    def save_meal_log(self, meal_data: Dict[str, Any]) -> Dict[str, Any]:
        if not self.token:
            return {"error": "Not authenticated", "status": "failed"}
        
        try:
            response = self.session.post(
                f"{self.service_url}/meals",
                json=meal_data
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e), "status": "failed"}
    
    def get_meal_history(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> List[Dict[str, Any]]:
        if not self.token:
            return [{"error": "Not authenticated", "status": "failed"}]
        
        params = {}
        if start_date:
            params["start"] = start_date
        if end_date:
            params["end"] = end_date
        
        try:
            response = self.session.get(
                f"{self.service_url}/meals",
                params=params
            )
            response.raise_for_status()
            return response.json().get("meals", [])
        except requests.exceptions.RequestException as e:
            return [{"error": str(e), "status": "failed"}]
