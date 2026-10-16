-- SpineGuard AI — Supabase PostgreSQL Schema
-- © 2026 Naresh (naresh-coderrr). All Rights Reserved.
-- Run this SQL in your Supabase Dashboard -> SQL Editor

CREATE TABLE IF NOT EXISTS public.app_users (
  id UUID PRIMARY KEY,
  username TEXT UNIQUE NOT NULL,
  email TEXT UNIQUE NOT NULL,
  password_hash TEXT NOT NULL,
  password_salt TEXT NOT NULL,
  password_fingerprint TEXT UNIQUE NOT NULL, -- Guarantees no two users ever share the same password!
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Enable Row Level Security (RLS)
ALTER TABLE public.app_users ENABLE ROW LEVEL SECURITY;

-- Allow backend / anon key operations for registration & unique-password check
DROP POLICY IF EXISTS "Allow public auth operations" ON public.app_users;
CREATE POLICY "Allow public auth operations" ON public.app_users
  FOR ALL USING (true) WITH CHECK (true);
