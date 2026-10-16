"""
AI Posture & Drowsiness Corrector — Auth & Unique Password Manager (Supabase + Local Fallback)
© 2026 Naresh (naresh-coderrr). All Rights Reserved.

Features:
- Enforces globally UNIQUE passwords across all users via deterministic SHA-256 fingerprint (`password_fingerprint`)
- Stores hashed passwords using PBKDF2-HMAC-SHA256 with per-user cryptographic salt
- Seamless runtime connection to Supabase (`users` and `study_sessions` tables) + local SQLite/JSON persistence fallback
"""

import os
import re
import json
import time
import uuid
import hashlib
import secrets
from typing import Dict, Any, Optional, Tuple

try:
    from supabase import create_client, Client
except ImportError:
    create_client = None
    Client = Any

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
LOCAL_DB_PATH = os.path.join(DATA_DIR, "users_store.json")
CONFIG_PATH = os.path.join(DATA_DIR, "supabase_config.json")


def hash_password_fingerprint(password: str) -> str:
    """
    Create a deterministic SHA-256 fingerprint of the password so we can enforce
    that NO two users ever share the same password in Supabase or local storage.
    """
    normalized = password.strip()
    return hashlib.sha256(f"spineguard_unique_pw_v1::{normalized}".encode("utf-8")).hexdigest()


def hash_password_secure(password: str, salt: Optional[str] = None) -> Tuple[str, str]:
    """Create a salted PBKDF2-HMAC-SHA256 password hash for storage."""
    if not salt:
        salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 120000)
    return dk.hex(), salt


def validate_password_strength(password: str) -> Tuple[bool, str]:
    """Validate password complexity requirements before checking global uniqueness."""
    if len(password) < 8:
        return False, "Password must be at least 8 characters long."
    if not re.search(r"[A-Z]", password):
        return False, "Password must include at least one uppercase letter (A-Z)."
    if not re.search(r"[a-z]", password):
        return False, "Password must include at least one lowercase letter (a-z)."
    if not re.search(r"[0-9]", password):
        return False, "Password must include at least one digit (0-9)."
    if not re.search(r"[^A-Za-z0-9]", password):
        return False, "Password must include at least one special character (!@#$%^&* etc.)."
    return True, "Strong password format."


class SupabaseAuthManager:
    def __init__(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        self.supabase_url: str = os.environ.get("SUPABASE_URL", "").strip()
        self.supabase_key: str = os.environ.get("SUPABASE_KEY", "").strip()
        self.client: Optional[Any] = None
        self.connection_error: Optional[str] = None

        self._load_saved_config()
        self._init_local_store()
        self._connect_supabase()

    def _load_saved_config(self):
        if os.path.exists(CONFIG_PATH):
            try:
                with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                    if not self.supabase_url and cfg.get("supabase_url"):
                        self.supabase_url = cfg["supabase_url"].strip()
                    if not self.supabase_key and cfg.get("supabase_key"):
                        self.supabase_key = cfg["supabase_key"].strip()
            except Exception:
                pass

    def _init_local_store(self):
        if not os.path.exists(LOCAL_DB_PATH):
            with open(LOCAL_DB_PATH, "w", encoding="utf-8") as f:
                json.dump({"users": [], "sessions": []}, f, indent=2)

    def _read_local(self) -> Dict[str, Any]:
        try:
            with open(LOCAL_DB_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"users": [], "sessions": []}

    def _write_local(self, data: Dict[str, Any]):
        with open(LOCAL_DB_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _connect_supabase(self) -> bool:
        self.client = None
        self.connection_error = None
        if not self.supabase_url or not self.supabase_key or not create_client:
            return False
        try:
            self.client = create_client(self.supabase_url, self.supabase_key)
            return True
        except Exception as e:
            self.connection_error = str(e)
            self.client = None
            return False

    def configure_supabase(self, url: str, key: str) -> Dict[str, Any]:
        """Connect or update Supabase credentials dynamically from the Web UI."""
        self.supabase_url = url.strip()
        self.supabase_key = key.strip()
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump({"supabase_url": self.supabase_url, "supabase_key": self.supabase_key}, f, indent=2)
        ok = self._connect_supabase()
        table_status = "disconnected"
        if ok and self.client:
            try:
                # Test querying app_users table
                self.client.table("app_users").select("id").limit(1).execute()
                table_status = "connected_and_verified"
            except Exception as e:
                err_str = str(e)
                if "app_users" in err_str or "relation" in err_str or "404" in err_str or "PGRST" in err_str:
                    table_status = "connected_needs_schema"
                    self.connection_error = "Connected to Supabase! Please run the provided SQL snippet in your Supabase SQL Editor to create the `app_users` table."
                else:
                    table_status = "error"
                    self.connection_error = err_str
        return self.get_status(extra_status=table_status)

    def get_status(self, extra_status: Optional[str] = None) -> Dict[str, Any]:
        local_data = self._read_local()
        return {
            "supabase_connected": self.client is not None,
            "supabase_url_masked": (
                self.supabase_url[:24] + "..." if len(self.supabase_url) > 24 else self.supabase_url
            ) if self.supabase_url else "",
            "status_detail": extra_status or ("connected" if self.client else "local_fallback_ready"),
            "connection_error": self.connection_error,
            "total_registered_users": len(local_data.get("users", [])),
        }

    def is_password_unique(self, password: str) -> Tuple[bool, str]:
        """
        Verify both password strength AND that no other user has ever used this password
        in Supabase (`app_users.password_fingerprint`) or local storage.
        """
        valid, msg = validate_password_strength(password)
        if not valid:
            return False, msg

        fp = hash_password_fingerprint(password)

        # 1. Check Supabase if connected
        if self.client:
            try:
                res = (
                    self.client.table("app_users")
                    .select("id, username")
                    .eq("password_fingerprint", fp)
                    .limit(1)
                    .execute()
                )
                if res.data and len(res.data) > 0:
                    return False, "This password is already taken by another user! Every user must have a 100% unique password."
            except Exception:
                pass

        # 2. Check Local Store
        local_data = self._read_local()
        for u in local_data.get("users", []):
            if u.get("password_fingerprint") == fp:
                return False, "This password is already taken by another user! Every user must have a 100% unique password."

        return True, "Password is strong and 100% unique across all users!"

    def register_user(self, username: str, email: str, password: str) -> Dict[str, Any]:
        username = username.strip()
        email = email.strip().lower()

        if len(username) < 3:
            return {"ok": False, "error": "Username must be at least 3 characters."}
        if "@" not in email or "." not in email:
            return {"ok": False, "error": "Please enter a valid email address."}

        # Check password uniqueness and strength
        is_unique, msg = self.is_password_unique(password)
        if not is_unique:
            return {"ok": False, "error": msg}

        local_data = self._read_local()
        for u in local_data.get("users", []):
            if u.get("email") == email:
                return {"ok": False, "error": "An account with this email already exists. Please log in."}
            if u.get("username", "").lower() == username.lower():
                return {"ok": False, "error": "This username is already taken."}

        fp = hash_password_fingerprint(password)
        pw_hash, salt = hash_password_secure(password)
        user_id = str(uuid.uuid4())
        created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        user_record = {
            "id": user_id,
            "username": username,
            "email": email,
            "password_hash": pw_hash,
            "password_salt": salt,
            "password_fingerprint": fp,
            "created_at": created_at,
        }

        supabase_saved = False
        supabase_note = "Saved to local database (Connect Supabase anytime to sync)."

        if self.client:
            try:
                # Check if email or username exists in Supabase
                existing = (
                    self.client.table("app_users")
                    .select("id, email, username")
                    .or_(f"email.eq.{email},username.eq.{username}")
                    .execute()
                )
                if existing.data and len(existing.data) > 0:
                    return {"ok": False, "error": "Email or username already exists in Supabase database."}

                self.client.table("app_users").insert(user_record).execute()
                supabase_saved = True
                supabase_note = "Stored securely in Supabase (`app_users`) with unique password constraint!"
            except Exception as e:
                supabase_note = f"Saved locally (Supabase sync note: {str(e)[:120]})"

        local_data["users"].append(user_record)
        self._write_local(local_data)

        token = secrets.token_urlsafe(32)
        return {
            "ok": True,
            "message": f"Account created! {supabase_note}",
            "supabase_saved": supabase_saved,
            "token": token,
            "user": {
                "id": user_id,
                "username": username,
                "email": email,
                "created_at": created_at,
                "supabase_synced": supabase_saved,
            },
        }

    def login_user(self, identifier: str, password: str) -> Dict[str, Any]:
        """Log in with email or username and unique password."""
        identifier = identifier.strip()
        fp = hash_password_fingerprint(password)

        # 1. Try Supabase first if connected
        if self.client:
            try:
                res = (
                    self.client.table("app_users")
                    .select("*")
                    .or_(f"email.eq.{identifier.lower()},username.eq.{identifier}")
                    .limit(1)
                    .execute()
                )
                if res.data and len(res.data) > 0:
                    u = res.data[0]
                    calc_hash, _ = hash_password_secure(password, u.get("password_salt", ""))
                    if calc_hash == u.get("password_hash") or fp == u.get("password_fingerprint"):
                        token = secrets.token_urlsafe(32)
                        return {
                            "ok": True,
                            "message": "Logged in via Supabase!",
                            "token": token,
                            "user": {
                                "id": u["id"],
                                "username": u["username"],
                                "email": u["email"],
                                "created_at": u.get("created_at"),
                                "supabase_synced": True,
                            },
                        }
                    else:
                        return {"ok": False, "error": "Invalid password for this account."}
            except Exception:
                pass

        # 2. Check Local Store
        local_data = self._read_local()
        for u in local_data.get("users", []):
            if u.get("email") == identifier.lower() or u.get("username", "").lower() == identifier.lower():
                calc_hash, _ = hash_password_secure(password, u.get("password_salt", ""))
                if calc_hash == u.get("password_hash") or fp == u.get("password_fingerprint"):
                    token = secrets.token_urlsafe(32)
                    return {
                        "ok": True,
                        "message": "Welcome back! Logged in successfully.",
                        "token": token,
                        "user": {
                            "id": u["id"],
                            "username": u["username"],
                            "email": u["email"],
                            "created_at": u.get("created_at"),
                            "supabase_synced": self.client is not None,
                        },
                    }
                else:
                    return {"ok": False, "error": "Incorrect password."}

        return {"ok": False, "error": "No account found with that email or username. Please sign up first!"}
