import asyncio
import secrets
import string
from werkzeug.security import generate_password_hash
from datetime import datetime
from config import users_collection

def generate_secure_password(length=12):
    """Generate a secure random password"""
    alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
    password = ''.join(secrets.choice(alphabet) for i in range(length))
    return password

async def create_admin_user():
    """Create initial admin user with secure random password"""
    # Check if ANY admin user already exists
    existing_admin = await users_collection.find_one({"username": "admin"})
    if existing_admin:
        print("❌ Admin user already exists!")
        print("⚠️  For security reasons, this script will not override existing admin.")
        print("💡 If you need to reset admin password, please do it manually through database.")
        return
    
    # Generate secure random password
    admin_password = generate_secure_password(16)
    hashed_password = generate_password_hash(admin_password)
    
    admin_user = {
        "username": "admin",
        "password": hashed_password,
        "created_at": datetime.utcnow(),
        "is_active": True,
        "role": "admin"
    }
    
    result = await users_collection.insert_one(admin_user)
    
    print("=" * 60)
    print("🎉 ADMIN USER CREATED SUCCESSFULLY!")
    print("=" * 60)
    print(f"📋 User ID: {result.inserted_id}")
    print(f"👤 Username: admin")
    print(f"🔐 Password: {admin_password}")
    print(f"🔒 Hash Method: werkzeug (pbkdf2:sha256)")
    print("=" * 60)
    print("⚠️  IMPORTANT SECURITY NOTES:")
    print("1. 📝 SAVE THIS PASSWORD IMMEDIATELY - it won't be shown again!")
    print("2. 🔄 Change this password after first login")
    print("3. 🗑️  Delete this script after use for security")
    print("4. 🚫 This script can only be run ONCE per database")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(create_admin_user())