"""
AI Posture & Drowsiness Corrector — Auth & Unique Password Manager (Supabase REST + Local Fallback)
© 2026 Naresh (naresh-coderrr). All Rights Reserved.

Features:
- Enforces globally UNIQUE passwords across all users via deterministic SHA-256 fingerprint (`password_fingerprint`)
- Stores hashed passwords using PBKDF2-HMAC-SHA256 with per-user cryptographic salt
- Native httpx PostgREST client supporting both Supabase `sb_publishable_...` keys and `eyJ...` JWT keys
"""

import os
import re
import json
import time
import uuid
import hashlib
import secrets
from typing import Dict, Any, Optional, Tuple, List

import httpx

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
        self.supabase_url: str = (
            os.environ.get("SUPABASE_URL")
            or os.environ.get("NEXT_PUBLIC_SUPABASE_URL")
            or "https://ohhmbexjraujmgbsszau.supabase.co"
        ).strip().rstrip("/")
        self.supabase_key: str = (
            os.environ.get("SUPABASE_KEY")
            or os.environ.get("NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY")
            or "sb_publishable_dzxCFv0sAmeE34YTmhgj4w_2felXVHm"
        ).strip()
        self.connected: bool = False
        self.table_verified: bool = False
        self.connection_error: Optional[str] = None

        self._load_saved_config()
        self._init_local_store()
        self._verify_supabase_connection()

    def _headers(self) -> Dict[str, str]:
        headers = {
            "apikey": self.supabase_key,
            "Content-Type": "application/json",
            "Prefer": "return=representation",
        }
        # Include Authorization Bearer if JWT format or publishable key
        headers["Authorization"] = f"Bearer {self.supabase_key}"
        return headers

    def _load_saved_config(self):
        if os.path.exists(CONFIG_PATH):
            try:
                with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                    if cfg.get("supabase_url"):
                        self.supabase_url = cfg["supabase_url"].strip().rstrip("/")
                    if cfg.get("supabase_key"):
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

    def _verify_supabase_connection(self) -> str:
        self.connected = False
        self.table_verified = False
        self.connection_error = None
        if not self.supabase_url or not self.supabase_key:
            return "disconnected"

        try:
            url = f"{self.supabase_url}/rest/v1/app_users?select=id&limit=1"
            with httpx.Client(timeout=6.0) as client:
                resp = client.get(url, headers=self._headers())
            if resp.status_code == 200:
                self.connected = True
                self.table_verified = True
                return "connected_and_verified"
            elif resp.status_code in (404, 400) or "PGRST" in resp.text or "app_users" in resp.text:
                # Project URL & API key are valid! Only the `app_users` table needs to be created in SQL Editor
                self.connected = True
                self.table_verified = False
                self.connection_error = (
                    "Connected to your Supabase project (ohhmbexjraujmgbsszau)! "
                    "Run the SQL schema once in your Supabase SQL Editor to create the `app_users` table."
                )
                return "connected_needs_schema"
            elif resp.status_code == 401:
                self.connected = False
                self.connection_error = f"Invalid Supabase API key (HTTP 401): {resp.text[:120]}"
                return "unauthorized"
            else:
                self.connected = True
                self.connection_error = f"Supabase responded with HTTP {resp.status_code}: {resp.text[:120]}"
                return "connected_with_warning"
        except Exception as e:
            self.connected = False
            self.connection_error = str(e)
            return "error"

    def configure_supabase(self, url: str, key: str) -> Dict[str, Any]:
        """Connect or update Supabase credentials dynamically from the Web UI."""
        self.supabase_url = url.strip().rstrip("/")
        self.supabase_key = key.strip()
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump({"supabase_url": self.supabase_url, "supabase_key": self.supabase_key}, f, indent=2)
        status_code = self._verify_supabase_connection()
        return self.get_status(extra_status=status_code)

    def get_status(self, extra_status: Optional[str] = None) -> Dict[str, Any]:
        local_data = self._read_local()
        return {
            "supabase_connected": self.connected,
            "table_verified": self.table_verified,
            "supabase_url_masked": (
                self.supabase_url[:32] + "..." if len(self.supabase_url) > 32 else self.supabase_url
            ) if self.supabase_url else "",
            "status_detail": extra_status or (
                "connected_and_verified" if self.table_verified else ("connected_needs_schema" if self.connected else "local_fallback_ready")
            ),
            "connection_error": self.connection_error,
            "total_registered_users": len(local_data.get("users", [])),
        }

    def _supabase_select(self, query_params: str) -> Optional[List[Dict[str, Any]]]:
        if not self.connected:
            return None
        try:
            url = f"{self.supabase_url}/rest/v1/app_users?{query_params}"
            with httpx.Client(timeout=5.0) as client:
                resp = client.get(url, headers=self._headers())
            if resp.status_code == 200:
                self.table_verified = True
                self.connection_error = None
                return resp.json()
        except Exception:
            pass
        return None

    def _supabase_insert(self, record: Dict[str, Any]) -> Tuple[bool, str]:
        if not self.connected:
            return False, "Supabase not connected"
        try:
            url = f"{self.supabase_url}/rest/v1/app_users"
            with httpx.Client(timeout=5.0) as client:
                resp = client.post(url, headers=self._headers(), json=record)
            if resp.status_code in (200, 201):
                self.table_verified = True
                self.connection_error = None
                return True, "Stored in Supabase (`app_users`)!"
            return False, f"HTTP {resp.status_code}: {resp.text[:140]}"
        except Exception as e:
            return False, str(e)

    def is_password_unique(self, password: str) -> Tuple[bool, str]:
        """
        Verify both password strength AND that no other user has ever used this password
        in Supabase (`app_users.password_fingerprint`) or local storage.
        """
        valid, msg = validate_password_strength(password)
        if not valid:
            return False, msg

        fp = hash_password_fingerprint(password)

        # 1. Check Supabase PostgREST if connected
        rows = self._supabase_select(f"select=id,username&password_fingerprint=eq.{fp}&limit=1")
        if rows and len(rows) > 0:
            return False, "This password is already taken by another user! Every user must have a 100% unique password."

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
        supabase_note = "Saved to local store."

        if self.connected:
            existing = self._supabase_select(
                f"select=id,email,username&or=(email.eq.{email},username.eq.{username})&limit=1"
            )
            if existing and len(existing) > 0:
                return {"ok": False, "error": "Email or username already exists in Supabase database."}

            ok_ins, ins_msg = self._supabase_insert(user_record)
            supabase_saved = ok_ins
            supabase_note = ins_msg

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

        # 1. Try Supabase PostgREST first if connected
        if self.connected:
            rows = self._supabase_select(
                f"select=*&or=(email.eq.{identifier.lower()},username.eq.{identifier})&limit=1"
            )
            if rows and len(rows) > 0:
                u = rows[0]
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
                            "supabase_synced": self.connected,
                        },
                    }
                else:
                    return {"ok": False, "error": "Incorrect password."}

        return {"ok": False, "error": "No account found with that email or username. Please sign up first!"}
