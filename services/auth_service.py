from typing import Optional
from datetime import datetime, timedelta
from werkzeug.security import generate_password_hash, check_password_hash
from jose import JWTError, jwt
from bson import ObjectId
from config import users_collection, SECRET_KEY, ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES
from models import UserCreate, UserUpdate, UserResponse, TokenData

class AuthService:
    def __init__(self):
        self.users_collection = users_collection

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash using werkzeug"""
        return check_password_hash(hashed_password, plain_password)

    def get_password_hash(self, password: str) -> str:
        """Hash a password using werkzeug"""
        return generate_password_hash(password)

    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None):
        """Create JWT access token"""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=15)
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt

    def verify_token(self, token: str) -> Optional[TokenData]:
        """Verify JWT token"""
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            username: str = payload.get("sub")
            if username is None:
                return None
            return TokenData(username=username)
        except JWTError:
            return None

    async def authenticate_user(self, username: str, password: str) -> Optional[dict]:
        """Authenticate user with username and password"""
        user = await self.users_collection.find_one({"username": username})
        if not user:
            return None
        if not self.verify_password(password, user["password"]):
            return None
        return user

    async def get_user_by_username(self, username: str) -> Optional[dict]:
        """Get user by username"""
        return await self.users_collection.find_one({"username": username})

    async def get_user_by_id(self, user_id: str) -> Optional[dict]:
        """Get user by ID"""
        try:
            return await self.users_collection.find_one({"_id": ObjectId(user_id)})
        except:
            return None

    async def create_user(self, user_data: UserCreate) -> UserResponse:
        """Create a new user"""
        # Check if username already exists
        existing_user = await self.get_user_by_username(user_data.username)
        if existing_user:
            raise ValueError("Username already exists")

        # Hash password
        hashed_password = self.get_password_hash(user_data.password)
        
        # Create user document
        user_doc = {
            "username": user_data.username,
            "password": hashed_password,
            "created_at": datetime.utcnow(),
            "is_active": True
        }
        
        # Insert user
        result = await self.users_collection.insert_one(user_doc)
        
        # Return user response
        return UserResponse(
            id=str(result.inserted_id),
            username=user_data.username,
            created_at=user_doc["created_at"],
            is_active=user_doc["is_active"]
        )

    async def get_all_users(self) -> list[UserResponse]:
        """Get all users"""
        users = []
        async for user in self.users_collection.find():
            users.append(UserResponse(
                id=str(user["_id"]),
                username=user["username"],
                created_at=user["created_at"],
                is_active=user.get("is_active", True)
            ))
        return users

    async def update_user(self, user_id: str, user_data: UserUpdate) -> Optional[UserResponse]:
        """Update user"""
        try:
            update_data = {}
            if user_data.password:
                update_data["password"] = self.get_password_hash(user_data.password)
            if user_data.is_active is not None:
                update_data["is_active"] = user_data.is_active
            
            if not update_data:
                return None
                
            result = await self.users_collection.update_one(
                {"_id": ObjectId(user_id)},
                {"$set": update_data}
            )
            
            if result.modified_count == 0:
                return None
                
            updated_user = await self.get_user_by_id(user_id)
            if updated_user:
                return UserResponse(
                    id=str(updated_user["_id"]),
                    username=updated_user["username"],
                    created_at=updated_user["created_at"],
                    is_active=updated_user.get("is_active", True)
                )
            return None
        except:
            return None

    async def delete_user(self, user_id: str) -> bool:
        """Delete user"""
        try:
            result = await self.users_collection.delete_one({"_id": ObjectId(user_id)})
            return result.deleted_count > 0
        except:
            return False