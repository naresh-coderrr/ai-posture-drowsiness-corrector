# 🛡️ SpineGuard AI — Real-Time Posture, Drowsiness & Study Distraction Corrector

> **Built with Python, MediaPipe Tasks (Face, Pose & Hand Landmarkers), FastAPI WebSockets, and Supabase Unique-Password Authentication.**  
> **© 2026 Naresh (`naresh-coderrr`). All Rights Reserved.**

---

## ✨ Key Features

1. **Real-Time Facial Aspect Ratio Tracking (Python + MediaPipe FaceLandmarker)**
   - **Eye Aspect Ratio (EAR):** Continuously computes 6-point landmark geometry for both left and right eyes:
     $$\text{EAR} = \frac{\|p_2 - p_6\| + \|p_3 - p_5\|}{2 \|p_1 - p_4\|}$$
   - **Blink Rate & Drowsiness Detection:** Tracks total eye blinks, blinks-per-minute (BPM), and triggers an instant critical wake-up alert if your eyes stay closed beyond 1.35 seconds.
   - **Mouth Aspect Ratio (MAR):** Detects heavy yawning and fatigue in real time.

2. **Live Spinal & Cervical Biomechanics (MediaPipe PoseLandmarker)**
   - Tracks nose-to-shoulder vertical compression (forward head / neck slouch), shoulder tilt asymmetry, lateral leaning, and head pitch/roll.
   - Includes 1-click **Calibrate Upright** baseline locks so it adapts to any webcam angle or desk setup.

3. **Mobile Phone & Study Distraction Sentinel (MediaPipe HandLandmarker + Head Pose)**
   - Detects when you pick up your **mobile phone** in your hand, rest your chin on your hand, or look away from your study screen.
   - Designed for **Background Study Tabs**: Uses an unthrottled Web Worker timer + silent Web Audio keep-alive stream + **Floating Picture-in-Picture (PiP) Study HUD** + **Laptop OS Desktop Notifications**, so you can switch to your study tabs (PDFs, YouTube lectures, coding IDE) while SpineGuard AI watches your posture and focus in the background!

4. **Toggleable Audio Alerts: Funny Drill-Sergeant Roasts vs. Tactical Wake Siren vs. Custom MP3**
   - **😂 Funny Roast Mode:** Plays comedic voice roasts (*"Whoa there shrimp posture! Sit up straight like a CEO!"*, *"Step away from the phone! Instagram Reels won't pass your exams!"*).
   - **🚨 Tactical Siren Mode:** High-visibility synthesized Web Audio siren.
   - **🎵 Custom Audio Upload:** Upload any `.mp3` or `.wav` file directly in the UI.

5. **5-Minute Free Trial + Supabase Globally Unique Password Authentication**
   - New visitors get an instant **5-Minute Free Trial** (`05:00` live countdown).
   - Once the 5-minute trial finishes, the app prompts the user to **Sign Up / Login**.
   - **100% Globally Unique Password Enforcement:** Every user's password generates a deterministic SHA-256 `password_fingerprint` (with `UNIQUE` constraint in Supabase PostgreSQL `app_users`) plus a salted `PBKDF2-HMAC-SHA256` hash—ensuring **no two users can ever share the same password**.

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
cd ai-posture-drowsiness-corrector
pip install -r requirements.txt
```

### 2. Run the Real-Time Server
```bash
python main.py
```
Then open **[http://localhost:8080](http://localhost:8080)** in your browser (Laptop or Mobile).  
*(Note: On first launch, `vision_engine.py` automatically downloads the official MediaPipe `.task` models into `./models/` if not already present).*

---

## ⚡ Connecting Your Supabase Database

You can connect your Supabase project either via environment variables (`SUPABASE_URL` and `SUPABASE_KEY`) **or live inside the Web UI** by clicking the **"Connect Supabase"** button in the top navigation bar!

Run the following SQL in your **Supabase SQL Editor** (also available to copy with 1 click inside the app's Supabase modal):

```sql
CREATE TABLE IF NOT EXISTS public.app_users (
  id UUID PRIMARY KEY,
  username TEXT UNIQUE NOT NULL,
  email TEXT UNIQUE NOT NULL,
  password_hash TEXT NOT NULL,
  password_salt TEXT NOT NULL,
  password_fingerprint TEXT UNIQUE NOT NULL,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

ALTER TABLE public.app_users ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Allow public auth operations" ON public.app_users
  FOR ALL USING (true) WITH CHECK (true);
```

---

## 📜 Copyright & License

**© 2026 Naresh (`naresh-coderrr`). All Rights Reserved.**  
Crafted with Python, MediaPipe, FastAPI, and Supabase.
