"""
Quick User Registration Script
Run this to create a test user in your database
"""
import os
from pymongo import MongoClient
from passlib.context import CryptContext
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def create_test_user():
    """Create a test user in the database"""
    
    # Connect to MongoDB
    mongodb_uri = os.getenv('MONGODB_URI')
    if not mongodb_uri:
        print("[ERROR] MONGODB_URI not found in environment variables")
        return
    
    client = MongoClient(mongodb_uri)
    db = client['first_aid_db']
    users_collection = db['users']
    
    # Test user credentials
    test_user = {
        'user_id': 'user_test_001',
        'email': 'test@example.com',
        'username': 'testuser',
        'hashed_password': pwd_context.hash('testpass123'),
        'full_name': 'Test User',
        'created_at': datetime.utcnow().isoformat(),
        'is_active': True
    }
    
    # Check if user exists
    existing = users_collection.find_one({'username': test_user['username']})
    if existing:
        print(f"[OK] User '{test_user['username']}' already exists")
        print(f"   Email: {test_user['email']}")
        print(f"   Password: testpass123")
        return
    
    # Create user
    try:
        users_collection.insert_one(test_user)
        print("[SUCCESS] Test user created successfully!")
        print(f"   Email: {test_user['email']}")
        print(f"   Username: {test_user['username']}")
        print(f"   Password: testpass123")
    except Exception as e:
        print(f"[ERROR] Error creating user: {e}")

if __name__ == "__main__":
    print("=" * 50)
    print("Creating Test User")
    print("=" * 50)
    create_test_user()
    print("=" * 50)
