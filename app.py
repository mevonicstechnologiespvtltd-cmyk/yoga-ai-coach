"""
Yoga AI Coach - Web Application
Run:  python app.py
Open: http://localhost:8080

How it works:
  1. ESP32 boots → calls /api/device-hello with its machine_id + IP (auto-registers)
  2. User opens web app → clicks "Add Device" → enters Machine ID + Password (no IP needed)
  3. App finds ESP32 IP from hello registry, stores device
  4. User chats: "I want to practice Padmasana" → AI monitors via camera → voice corrections
"""

import os, json, time, threading, io, wave, base64, hashlib
import numpy as np
import requests
from flask import Flask, request, jsonify, Response
import socket

app = Flask(__name__)

# ─── State ───────────────────────────────────────────────────────────────────
_DIR        = os.path.dirname(os.path.abspath(__file__))
_STATE_FILE = os.path.join(_DIR, "devices.json")

_devices        = {}   # {machine_id: {name, ip, port, pw_hash, added_at}}
_hello_registry = {}   # {machine_id: {ip, port, last_seen}}  — self-registered by ESP32
_sessions       = {}   # {session_id: {device_id, messages, asana, monitoring}}
_state_lock     = threading.Lock()

# Gemini API key — set via GEMINI_API_KEY env var (or update via Settings in the UI)
_gemini_key = os.environ.get("GEMINI_API_KEY", "")

# Gemini model — multimodal, so one model handles vision + chat. Update here if Google deprecates it.
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
GEMINI_URL   = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"


def _llm(messages, max_tokens, temperature) -> str:
    """Gemini via its OpenAI-compatible endpoint."""
    r = requests.post(
        GEMINI_URL,
        headers={"Authorization": f"Bearer {_gemini_key}"},
        json={"model": GEMINI_MODEL, "messages": messages, "max_tokens": max_tokens,
              "temperature": temperature, "reasoning_effort": "none"},
        timeout=60,
    )
    d = r.json()
    if isinstance(d, list):
        d = d[0]
    if r.status_code != 200:
        raise RuntimeError(d.get("error", {}).get("message", r.text[:200]))
    return d["choices"][0]["message"]["content"] or ""


def _load():
    global _devices, _gemini_key
    if os.path.exists(_STATE_FILE):
        try:
            d = json.load(open(_STATE_FILE, encoding="utf-8"))
            _devices.update(d.get("devices", {}))
            if d.get("gemini_key"):        # user-updated key takes priority
                _gemini_key = d["gemini_key"]
        except Exception as e:
            print(f"[State] Load error: {e}")


def _save():
    try:
        with open(_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump({"devices": _devices, "gemini_key": _gemini_key}, f, indent=2)
    except Exception as e:
        print(f"[State] Save error: {e}")


_load()


# ─── TTS (Deepgram → mono 16-bit WAV → ESP32 speaker) ────────────────────────
DEEPGRAM_KEY = os.environ.get("DEEPGRAM_API_KEY", "")
_tts_enabled = True


def _to_mono16(wav_bytes: bytes) -> bytes:
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            ch, sw, rate = wf.getnchannels(), wf.getsampwidth(), wf.getframerate()
            raw = wf.readframes(wf.getnframes())
        if ch == 1 and sw == 2:
            return wav_bytes
        if sw == 1:
            s = ((np.frombuffer(raw, np.uint8).astype(np.int16)) - 128) * 256
        elif sw == 2:
            s = np.frombuffer(raw, np.int16).copy()
        elif sw == 4:
            s = (np.frombuffer(raw, np.int32) // 65536).astype(np.int16)
        else:
            return wav_bytes
        if ch > 1:
            s = s.reshape(-1, ch).mean(axis=1).astype(np.int16)
        out = io.BytesIO()
        with wave.open(out, "wb") as wf:
            wf.setnchannels(1); wf.setsampwidth(2)
            wf.setframerate(rate); wf.writeframes(s.tobytes())
        return out.getvalue()
    except Exception as e:
        print(f"[TTS] WAV convert error: {e}")
        return wav_bytes


def deepgram_tts(text: str) -> bytes:
    """Call Deepgram TTS, return mono 16-bit WAV bytes for MAX98357A."""
    if not text.strip():
        return b""
    try:
        r = requests.post(
            "https://api.deepgram.com/v1/speak",
            params={"model": "aura-asteria-en", "encoding": "linear16",
                    "sample_rate": 16000, "container": "wav"},
            headers={"Authorization": f"Token {DEEPGRAM_KEY}",
                     "Content-Type": "application/json"},
            json={"text": text},
            timeout=15,
        )
        if r.status_code == 200 and len(r.content) > 44:
            return _to_mono16(r.content)
        print(f"[Deepgram] Error {r.status_code}: {r.text[:120]}")
    except Exception as ex:
        print(f"[Deepgram] {ex}")
    return b""


# ─── Frame cache — populated by the stream proxy, read by _snapshot() ─────────
# This avoids opening a second connection to the ESP32 (it can only handle one)
_frame_cache:  dict = {}   # {machine_id: bytes}  — latest JPEG pushed by ESP32
_audio_queues: dict = {}   # {machine_id: bytes}  — WAV waiting for ESP32 to pull
_frame_lock = threading.Lock()


# ─── Device helpers ──────────────────────────────────────────────────────────
def _hash(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()


def _snapshot(mid: str) -> bytes | None:
    with _frame_lock:
        return _frame_cache.get(mid)


def _play_on_device(mid: str, wav: bytes):
    if not wav: return
    _audio_queues[mid] = wav
    print(f"[Audio] Queued {len(wav)} B for '{mid}' — ESP32 will pull on next poll")


def _speak(text: str, mid: str):
    if not _tts_enabled:
        return
    print(f"[Voice] {text[:90]}")
    wav = deepgram_tts(text)
    if wav:
        _play_on_device(mid, wav)


def _speak_async(text: str, mid: str):
    if not _tts_enabled:
        return
    threading.Thread(target=_speak, args=(text, mid), daemon=True).start()


# ─── Active monitoring ────────────────────────────────────────────────────────
_mon = {"sid": None, "stop": None, "thread": None}
_mon_lock = threading.Lock()


def _stop_monitoring():
    with _mon_lock:
        if _mon["stop"]:
            _mon["stop"].set()
        _mon["sid"] = None


def _monitor_worker(sid: str, mid: str, asana: str, stop_evt: threading.Event):
    try:
        _speak(f"Starting {asana}. Please get into position in front of the camera.", mid)
        stop_evt.wait(3)

        while not stop_evt.is_set():
            frame = _snapshot(mid)
            if not frame:
                stop_evt.wait(3)
                continue

            b64 = base64.b64encode(frame).decode()
            try:
                correction = _llm(
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": (
                                f"I am practicing the {asana} yoga pose. "
                                "Look at my body in this image carefully. "
                                "Give me ONE specific correction or encouragement in 1-2 short sentences. "
                                "Be direct, kind, and helpful. Plain text only, no markdown."
                            )},
                            {"type": "image_url",
                             "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                        ]
                    }],
                    max_tokens=100,
                    temperature=0.3,
                ).strip()
                print(f"[Monitor] {correction[:80]}")

                with _state_lock:
                    if sid in _sessions:
                        _sessions[sid]["messages"].append({
                            "role": "assistant",
                            "content": correction,
                            "type": "correction",
                            "ts": time.time()
                        })

                _speak(correction, mid)

            except Exception as e:
                print(f"[Vision] {e}")

            stop_evt.wait(8)

    except Exception as e:
        print(f"[Monitor] Worker error: {e}")
    finally:
        with _state_lock:
            if sid in _sessions:
                _sessions[sid]["monitoring"] = False
        _speak("Monitoring stopped. Great practice!", mid)


# ─── API: Device hello (ESP32 calls this on boot to say it's online) ──────────
@app.route("/api/device-hello", methods=["POST"])
def api_device_hello():
    d   = request.get_json() or {}
    mid = d.get("machine_id", "").strip()
    if not mid:
        return jsonify({"ok": False}), 400
    _hello_registry[mid] = {"last_seen": time.time()}
    print(f"[Hello] '{mid}' online")
    if mid in _devices:
        _save()
    return jsonify({"ok": True})


# ─── API: Settings ────────────────────────────────────────────────────────────
@app.route("/api/settings", methods=["GET", "POST"])
def api_settings():
    global _gemini_key
    if request.method == "POST":
        d = request.get_json() or {}
        if "gemini_key" in d:
            _gemini_key = d["gemini_key"].strip()
            _save()
        return jsonify({"ok": True})
    partial = (_gemini_key[:8] + "...") if len(_gemini_key) > 8 else ""
    return jsonify({"set": bool(_gemini_key), "partial": partial})


# ─── API: Devices ─────────────────────────────────────────────────────────────
@app.route("/api/devices")
def api_devices_list():
    now = time.time()
    result = []
    for mid, d in _devices.items():
        hr = _hello_registry.get(mid, {})
        online = (now - hr.get("last_seen", 0)) < 120
        result.append({
            "id": mid, "name": d["name"],
            "added": d.get("added_at", ""), "online": online
        })
    return jsonify(result)


@app.route("/api/devices/add", methods=["POST"])
def api_devices_add():
    d   = request.get_json() or {}
    mid = d.get("machine_id", "").strip()
    pw  = d.get("password", "").strip()
    nm  = d.get("name", "").strip() or mid

    if not mid or not pw:
        return jsonify({"error": "Machine ID and password are required"}), 400

    # Accept if device recently sent a hello OR is actively pushing frames
    hello = _hello_registry.get(mid)
    has_frames = mid in _frame_cache
    if not hello and not has_frames:
        return jsonify({
            "error": f"Device '{mid}' has not connected yet. "
                     f"Power on the ESP32 and wait a few seconds, then try again."
        }), 400

    if hello:
        age = time.time() - hello["last_seen"]
        if age > 300 and not has_frames:
            return jsonify({
                "error": f"Device '{mid}' last seen {int(age/60)} min ago. "
                         f"Restart the ESP32 so it reconnects."
            }), 400

    _devices[mid] = {
        "name": nm, "pw_hash": _hash(pw),
        "added_at": time.strftime("%Y-%m-%d %H:%M")
    }
    _save()
    return jsonify({"ok": True, "id": mid, "name": nm})


@app.route("/api/devices/<mid>", methods=["DELETE"])
def api_devices_del(mid):
    _devices.pop(mid, None)
    _save()
    return jsonify({"ok": True})


# ─── API: Camera stream proxy ─────────────────────────────────────────────────
@app.route("/api/stream/<mid>")
def api_stream(mid):
    if mid not in _devices:
        return jsonify({"error": "device not found"}), 404

    def gen():
        last = None
        while True:
            with _frame_lock:
                frame = _frame_cache.get(mid)
            if frame and frame is not last:
                last = frame
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            time.sleep(0.033)

    return Response(gen(), content_type="multipart/x-mixed-replace; boundary=frame",
                    headers={"Cache-Control": "no-cache", "Access-Control-Allow-Origin": "*"})


@app.route("/api/snapshot/<mid>")
def api_snapshot_route(mid):
    frame = _snapshot(mid)
    if not frame:
        return "", 503
    return Response(frame, content_type="image/jpeg",
                    headers={"Cache-Control": "no-cache"})


# ─── API: Chat sessions ───────────────────────────────────────────────────────
@app.route("/api/chat/new", methods=["POST"])
def api_chat_new():
    d   = request.get_json() or {}
    mid = d.get("device_id", "") or None
    sid = str(int(time.time() * 1000))
    _stop_monitoring()
    _sessions[sid] = {
        "device_id": mid,
        "messages":  [],
        "asana":     None,
        "monitoring": False,
    }
    return jsonify({"session_id": sid})


@app.route("/api/chat/<sid>/messages")
def api_chat_msgs(sid):
    if sid not in _sessions:
        return jsonify([])
    return jsonify(_sessions[sid]["messages"])


@app.route("/api/chat/<sid>/status")
def api_chat_status(sid):
    if sid not in _sessions:
        return jsonify({"monitoring": False, "asana": None})
    s = _sessions[sid]
    return jsonify({
        "monitoring":  s.get("monitoring", False),
        "asana":       s.get("asana"),
        "device_id":   s.get("device_id"),
        "msg_count":   len(s.get("messages", []))
    })


@app.route("/api/chat/<sid>/send", methods=["POST"])
def api_chat_send(sid):
    if sid not in _sessions:
        return jsonify({"error": "session not found"}), 404

    data = request.get_json() or {}
    msg  = data.get("message", "").strip()
    if not msg:
        return jsonify({"error": "empty message"}), 400

    session = _sessions[sid]
    mid     = session.get("device_id")

    with _state_lock:
        session["messages"].append({
            "role": "user", "content": msg,
            "type": "chat", "ts": time.time()
        })

    if not _gemini_key:
        reply = "Gemini API key is not configured. Please set it in Settings."
        session["messages"].append({"role": "assistant", "content": reply,
                                    "type": "chat", "ts": time.time()})
        return jsonify({"reply": reply})

    # Capture a frame if device is connected — use vision model for every chat message
    frame_b64 = None
    if mid:
        frame = _snapshot(mid)
        if frame:
            frame_b64 = base64.b64encode(frame).decode()

    use_vision = frame_b64 is not None

    if use_vision:
        system = (
            "You are Yoga AI Coach with a live camera feed of the user.\n"
            "An image from their camera is attached to this message — look at it carefully.\n"
            "Describe what you actually see: background colour, objects, the person's posture and position.\n"
            "Use what you see to give accurate, specific yoga guidance.\n"
            "RULE: When the user wants to PRACTICE a yoga pose, respond ONLY with this JSON:\n"
            '{"action":"start","asana":"<Sanskrit name>","msg":"<one sentence>"}\n'
            "RULE: When the user says stop/done/quit, respond ONLY with:\n"
            '{"action":"stop","msg":"<one sentence>"}\n'
            "Otherwise reply naturally based on what you see. Under 3 sentences. Plain text only."
        )
    else:
        system = (
            "You are Yoga AI Coach, a friendly expert yoga instructor.\n"
            "RULE: When the user wants to PRACTICE a yoga pose, respond ONLY with this JSON:\n"
            '{"action":"start","asana":"<Sanskrit name>","msg":"<one sentence>"}\n'
            "RULE: When the user says stop/done/quit, respond ONLY with:\n"
            '{"action":"stop","msg":"<one sentence>"}\n'
            "Otherwise reply naturally as a yoga instructor. Under 3 sentences. Plain text only."
        )

    # Build history (text-only for past messages — vision API only supports image in latest message)
    history = [{"role": "system", "content": system}]
    for m in session["messages"][-12:]:
        if m["role"] in ("user", "assistant") and m.get("type") == "chat":
            history.append({"role": m["role"], "content": str(m["content"])})

    # Build current user message — include live frame if available
    if use_vision:
        current_user_msg = {
            "role": "user",
            "content": [
                {"type": "text", "text": msg},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"}}
            ]
        }
        # Replace the last user entry in history with the vision version
        if history and history[-1]["role"] == "user":
            history[-1] = current_user_msg
        else:
            history.append(current_user_msg)
    # (text-only path: history already has the user message as plain text)

    try:
        raw = _llm(history, max_tokens=220, temperature=0.6).strip()

        action = None
        try:
            clean = raw.strip("` \n")
            if clean.startswith("json"):
                clean = clean[4:].strip()
            if clean.startswith("{"):
                action = json.loads(clean)
        except Exception:
            pass

        if action and action.get("action") == "start":
            asana = action.get("asana", "yoga pose")
            reply = action.get("msg", f"Starting {asana} monitoring!")
            session["asana"] = asana

            if mid:
                _stop_monitoring()
                stop_evt = threading.Event()
                with _mon_lock:
                    _mon["sid"]  = sid
                    _mon["stop"] = stop_evt
                    t = threading.Thread(
                        target=_monitor_worker,
                        args=(sid, mid, asana, stop_evt),
                        daemon=True
                    )
                    _mon["thread"] = t
                with _state_lock:
                    session["monitoring"] = True
                t.start()
                reply = reply + " I'll watch your form and give live corrections through the speaker."
                _speak_async(reply, mid)
            else:
                reply = reply + " (Add an ESP32-CAM in the sidebar to get live pose monitoring.)"

            session["messages"].append({"role": "assistant", "content": reply,
                                        "type": "chat", "ts": time.time()})
            return jsonify({"reply": reply, "action": "start", "asana": asana})

        elif action and action.get("action") == "stop":
            reply = action.get("msg", "Stopping. Great practice!")
            _stop_monitoring()
            session["monitoring"] = False
            if mid:
                _speak_async(reply, mid)
            session["messages"].append({"role": "assistant", "content": reply,
                                        "type": "chat", "ts": time.time()})
            return jsonify({"reply": reply, "action": "stop"})

        else:
            if mid:
                _speak_async(raw, mid)
            session["messages"].append({"role": "assistant", "content": raw,
                                        "type": "chat", "ts": time.time()})
            return jsonify({"reply": raw})

    except Exception as e:
        err = f"AI error: {e}"
        print(f"[Chat] {e}")
        session["messages"].append({"role": "assistant", "content": err,
                                    "type": "chat", "ts": time.time()})
        return jsonify({"reply": err, "error": str(e)})


@app.route("/api/chat/<sid>/stop", methods=["POST"])
def api_chat_stop(sid):
    _stop_monitoring()
    if sid in _sessions:
        _sessions[sid]["monitoring"] = False
    return jsonify({"ok": True})


# ─── API: TTS toggle ──────────────────────────────────────────────────────────
@app.route("/api/tts", methods=["GET", "POST"])
def api_tts():
    global _tts_enabled
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        _tts_enabled = bool(data.get("enabled", not _tts_enabled))
    return jsonify({"enabled": _tts_enabled})


# ─── API: ESP32 push / pull ───────────────────────────────────────────────────
@app.route("/api/push-frame/<mid>", methods=["POST"])
def api_push_frame(mid):
    frame = request.data
    if len(frame) > 100:
        with _frame_lock:
            _frame_cache[mid] = frame
    return Response(b"", status=204)


@app.route("/api/pull-audio/<mid>")
def api_pull_audio(mid):
    wav = _audio_queues.pop(mid, None)
    if wav:
        return Response(wav, mimetype="audio/wav",
                        headers={"Content-Length": str(len(wav))})
    return Response(b"", status=204)


# ─── Debug: full TTS pipeline test ────────────────────────────────────────────
@app.route("/api/tts-test/<mid>")
def api_tts_test(mid):
    steps = []
    steps.append(f"tts_enabled={_tts_enabled}")
    d = _devices.get(mid)
    if not d:
        return jsonify({"steps": steps, "error": f"device '{mid}' not in _devices"})
    ip, port = d["ip"], d.get("port", 80)
    steps.append(f"device ip={ip}:{port}")
    try:
        r = requests.post(
            "https://api.deepgram.com/v1/speak",
            params={"model": "aura-asteria-en", "encoding": "linear16",
                    "sample_rate": 16000, "container": "wav"},
            headers={"Authorization": f"Token {DEEPGRAM_KEY}",
                     "Content-Type": "application/json"},
            json={"text": "Testing speaker. Can you hear me?"},
            timeout=15,
        )
        steps.append(f"deepgram={r.status_code} bytes={len(r.content)}")
        if r.status_code != 200:
            return jsonify({"steps": steps, "error": f"deepgram error: {r.text[:200]}"})
        wav = _to_mono16(r.content)
        steps.append(f"wav_after_convert={len(wav)} bytes")
        r2 = requests.post(
            f"http://{ip}:{port}/play", data=wav,
            headers={"Content-Type": "audio/wav"}, timeout=15
        )
        steps.append(f"esp32_play={r2.status_code}")
        return jsonify({"steps": steps, "ok": True})
    except Exception as ex:
        steps.append(f"exception={ex}")
        return jsonify({"steps": steps, "error": str(ex)})


# ─── API: TTS audio for browser playback ─────────────────────────────────────
@app.route("/api/speak", methods=["POST"])
def api_speak():
    if not _tts_enabled:
        return Response(b"", status=204)
    text = (request.get_json(silent=True) or {}).get("text", "").strip()
    if not text:
        return Response(b"", status=204)
    try:
        r = requests.post(
            "https://api.deepgram.com/v1/speak",
            params={"model": "aura-asteria-en"},
            headers={"Authorization": f"Token {DEEPGRAM_KEY}",
                     "Content-Type": "application/json"},
            json={"text": text},
            timeout=15,
        )
        if r.status_code == 200 and r.content:
            return Response(r.content, mimetype="audio/mpeg")
        print(f"[Speak] Deepgram {r.status_code}: {r.text[:80]}")
    except Exception as ex:
        print(f"[Speak] {ex}")
    return Response(b"", status=204)


# ─── Server info ──────────────────────────────────────────────────────────────
@app.route("/api/server-info")
def api_server_info():
    try:
        ip = socket.gethostbyname(socket.gethostname())
    except Exception:
        ip = "127.0.0.1"
    return jsonify({"ip": ip, "port": 8080})


# ─── Main HTML page ───────────────────────────────────────────────────────────
_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Yoga AI Coach</title>
<style>
:root{
  --bg:#0e0e1a;--card:#161628;--border:#252545;
  --accent:#7ec8b4;--text:#dde0f0;--dim:#5a6080;
  --danger:#d96060;--ok:#60d890;--warn:#d9c060;
  --ubub:#16305a;--abub:#0f1f12;--cbub:#231a05;
}
*{margin:0;padding:0;box-sizing:border-box}
body{background:var(--bg);color:var(--text);font-family:'Segoe UI',system-ui,sans-serif;
  height:100vh;display:flex;flex-direction:column;overflow:hidden}

.hdr{display:flex;align-items:center;padding:10px 18px;background:var(--card);
  border-bottom:1px solid var(--border);gap:10px;flex-shrink:0}
.hdr-title{font-size:17px;font-weight:700;color:var(--accent);letter-spacing:3px;flex:1}
.hdr-ip{font-size:11px;color:var(--dim);background:var(--bg);padding:4px 10px;
  border-radius:8px;border:1px solid var(--border)}
.hbtn{background:none;border:1px solid var(--border);color:var(--dim);
  padding:6px 14px;border-radius:20px;cursor:pointer;font-size:12px;transition:.2s}
.hbtn:hover{border-color:var(--accent);color:var(--accent)}
.tts-on{border-color:var(--accent)!important;color:var(--accent)!important}
.tts-off{border-color:var(--dim);color:var(--dim)}

.body{display:flex;flex:1;overflow:hidden}

/* Sidebar */
.sidebar{width:200px;border-right:1px solid var(--border);
  display:flex;flex-direction:column;padding:12px 10px;gap:6px;overflow-y:auto;flex-shrink:0}
.sb-title{font-size:10px;letter-spacing:2px;color:var(--dim);text-transform:uppercase;margin-bottom:2px}
.dev-item{background:var(--card);border:1px solid var(--border);border-radius:10px;
  padding:9px 11px;cursor:pointer;transition:.15s;position:relative}
.dev-item:hover{border-color:var(--accent)}
.dev-item.active{border-color:var(--accent);background:#162420}
.dev-name{font-size:13px;font-weight:600;color:var(--text);padding-right:18px}
.dev-ip{font-size:11px;color:var(--dim);margin-top:2px}
.dev-badge{display:inline-block;width:7px;height:7px;border-radius:50%;
  margin-right:5px;vertical-align:middle}
.dev-badge.on{background:var(--ok)}
.dev-badge.off{background:var(--dim)}
.dev-del{position:absolute;top:8px;right:8px;background:none;border:none;
  color:var(--danger);cursor:pointer;font-size:12px;opacity:.5;padding:0 2px}
.dev-del:hover{opacity:1}
.add-btn{background:none;border:1px dashed var(--border);color:var(--dim);
  padding:8px;border-radius:10px;cursor:pointer;font-size:12px;
  text-align:center;width:100%;transition:.2s;margin-top:2px}
.add-btn:hover{border-color:var(--accent);color:var(--accent)}
.no-dev{color:var(--dim);font-size:12px;text-align:center;padding:16px 6px;line-height:1.7}

/* Camera */
.cam-col{width:290px;border-right:1px solid var(--border);
  display:flex;flex-direction:column;padding:12px;gap:10px;flex-shrink:0}
.col-title{font-size:10px;letter-spacing:2px;color:var(--dim);text-transform:uppercase}
#cam-img{width:100%;border-radius:12px;border:1px solid var(--border);display:none}
.no-cam{width:100%;aspect-ratio:4/3;background:#09091a;border-radius:12px;
  border:1px solid var(--border);display:flex;align-items:center;
  justify-content:center;color:var(--dim);font-size:12px;text-align:center;line-height:1.8}
.stat-box{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:10px 12px}
.stat-lbl{font-size:10px;color:var(--dim);text-transform:uppercase;letter-spacing:1px}
.stat-val{font-size:14px;font-weight:600;color:var(--accent);margin-top:3px;
  display:flex;align-items:center;gap:6px}
.dot{width:7px;height:7px;border-radius:50%;background:var(--dim);flex-shrink:0}
.dot.on{background:var(--ok);box-shadow:0 0 6px var(--ok);animation:blink 1.4s infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.3}}

/* Chat */
.chat-col{flex:1;display:flex;flex-direction:column;padding:12px;gap:10px;overflow:hidden;min-width:0}
.msgs{flex:1;overflow-y:auto;display:flex;flex-direction:column;gap:7px;padding-right:2px}
.hint{text-align:center;color:var(--dim);font-size:12px;padding:18px 0;line-height:1.9}
.hint em{color:var(--accent)}
.hint strong{color:var(--text)}

.bub{max-width:88%;padding:9px 13px;border-radius:14px;font-size:13px;
  line-height:1.55;animation:fin .25s ease}
@keyframes fin{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:translateY(0)}}
.bub.u{background:var(--ubub);color:var(--text);align-self:flex-end;border-bottom-right-radius:4px}
.bub.a{background:var(--abub);color:var(--text);align-self:flex-start;
  border:1px solid var(--border);border-bottom-left-radius:4px}
.bub.c{background:var(--cbub);color:#f5d878;align-self:flex-start;
  border:1px solid #3a2f08;border-bottom-left-radius:4px}
.bub-lbl{font-size:9px;color:var(--dim);text-transform:uppercase;letter-spacing:1px;margin-bottom:3px}
.bub.c .bub-lbl{color:#8a7030}

.inp-row{display:flex;gap:7px;align-items:flex-end;flex-shrink:0}
textarea.ci{flex:1;background:var(--card);border:1px solid var(--border);color:var(--text);
  padding:9px 13px;border-radius:18px;font-size:13px;
  resize:none;min-height:40px;max-height:90px;outline:none;font-family:inherit}
textarea.ci:focus{border-color:var(--accent)}
.sbtn{background:var(--accent);color:#0a0a16;border:none;padding:9px 16px;
  border-radius:18px;cursor:pointer;font-weight:700;font-size:13px;white-space:nowrap;transition:.15s}
.sbtn:hover{opacity:.88}
.stopbtn{background:var(--danger);color:#fff;border:none;padding:9px 14px;
  border-radius:18px;cursor:pointer;font-weight:700;font-size:12px;white-space:nowrap;display:none}
.stopbtn.show{display:block}

/* Modals */
.backdrop{position:fixed;inset:0;background:rgba(0,0,0,.75);z-index:200;
  display:none;align-items:center;justify-content:center}
.backdrop.open{display:flex}
.modal{background:var(--card);border:1px solid var(--border);border-radius:16px;
  padding:22px;width:90%;max-width:400px}
.modal h2{font-size:16px;color:var(--accent);margin-bottom:6px}
.modal-sub{font-size:12px;color:var(--dim);margin-bottom:16px;line-height:1.5}
.fg{margin-bottom:12px}
.fg label{display:block;font-size:10px;color:var(--dim);text-transform:uppercase;
  letter-spacing:1px;margin-bottom:4px}
.fg input{width:100%;background:var(--bg);border:1px solid var(--border);color:var(--text);
  padding:8px 11px;border-radius:8px;font-size:13px;outline:none}
.fg input:focus{border-color:var(--accent)}
.fg .hint2{font-size:11px;color:var(--dim);margin-top:4px}
.mbtns{display:flex;gap:8px;justify-content:flex-end;margin-top:16px}
.mbtn{padding:8px 18px;border-radius:16px;border:none;cursor:pointer;font-size:13px;
  font-weight:600;transition:.15s}
.mbtn.p{background:var(--accent);color:#0a0a16}
.mbtn.c{background:none;border:1px solid var(--border);color:var(--dim)}
.mbtn:hover{opacity:.85}
.merr{color:var(--danger);font-size:12px;margin-top:8px;display:none;line-height:1.5}
.minfo{color:var(--dim);font-size:12px;margin-top:6px}

/* Waiting badge */
.waiting-badge{display:inline-block;background:#2a2a0a;color:var(--warn);
  border:1px solid #4a4a10;border-radius:8px;padding:3px 8px;font-size:11px;margin-top:4px}

::-webkit-scrollbar{width:3px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px}
</style>
</head>
<body>

<div class="hdr">
  <span class="hdr-title">&#129496; YOGA AI COACH</span>
  <span class="hdr-ip" id="srv-ip" title="Set this as SERVER_IP in ESP32 firmware"></span>
  <button class="hbtn tts-on" id="tts-btn" onclick="toggleTTS()" title="Toggle voice output on ESP32 speaker">&#128266; Voice ON</button>
  <button class="hbtn" onclick="openM('sm')">&#9881; Settings</button>
</div>

<div class="body">

  <!-- Sidebar -->
  <div class="sidebar">
    <div class="sb-title">My Devices</div>
    <div id="devlist"><div class="no-dev">No devices yet</div></div>
    <button class="add-btn" onclick="openM('adm')">&#43; Add Device</button>
  </div>

  <!-- Camera -->
  <div class="cam-col">
    <div class="col-title">Live Camera</div>
    <div id="nocam" class="no-cam">Select a device<br>to see live feed</div>
    <img id="cam-img" alt="Live camera feed">
    <div class="stat-box">
      <div class="stat-lbl">Status</div>
      <div class="stat-val"><span class="dot" id="sdot"></span><span id="stxt">Idle</span></div>
    </div>
    <div class="stat-box" id="asana-box" style="display:none">
      <div class="stat-lbl">Monitoring Pose</div>
      <div class="stat-val" id="asana-val" style="font-size:13px;flex-wrap:wrap"></div>
    </div>
  </div>

  <!-- Chat -->
  <div class="chat-col">
    <div class="col-title">AI Chat</div>
    <div class="msgs" id="msgs">
      <div class="hint">
        &#128075; Welcome to Yoga AI Coach!<br><br>
        <strong>Quick start:</strong><br>
        1&#65039;&#8419; Power on your ESP32-CAM<br>
        2&#65039;&#8419; Click <em>+ Add Device</em> in the sidebar<br>
        3&#65039;&#8419; Select the device &amp; start chatting<br><br>
        Try: <em>"I want to practice Padmasana"</em>
      </div>
    </div>
    <div class="inp-row">
      <textarea class="ci" id="ci" placeholder="Ask your AI yoga coach anything..." rows="1"
        onkeydown="handleKey(event)" oninput="autoH(this)"></textarea>
      <button class="sbtn" onclick="sendMsg()">Send</button>
      <button class="stopbtn" id="stopbtn" onclick="stopMon()">&#9632; Stop</button>
    </div>
  </div>
</div>

<!-- Settings modal -->
<div class="backdrop" id="sm">
  <div class="modal">
    <h2>&#9881; Settings</h2>
    <div class="fg">
      <label>Gemini API Key</label>
      <input type="password" id="gk" placeholder="AIza... / AQ...">
    </div>
    <div class="minfo" id="kinfo"></div>
    <div class="mbtns">
      <button class="mbtn c" onclick="closeM('sm')">Cancel</button>
      <button class="mbtn p" onclick="saveSettings()">Save</button>
    </div>
  </div>
</div>

<!-- Add Device modal -->
<div class="backdrop" id="adm">
  <div class="modal">
    <h2>&#128247; Add ESP32-CAM</h2>
    <p class="modal-sub">
      Make sure your ESP32-CAM is powered on and connected to WiFi.<br>
      The device registers itself automatically when it boots.
    </p>
    <div class="fg">
      <label>Device Name <span style="color:var(--dim)">(optional label)</span></label>
      <input type="text" id="dnm" placeholder="My Yoga Camera">
    </div>
    <div class="fg">
      <label>Machine ID</label>
      <input type="text" id="dmid" placeholder="yoga-cam-1">
      <div class="hint2">Must match the MACHINE_ID in your ESP32 firmware</div>
    </div>
    <div class="fg">
      <label>Password</label>
      <input type="password" id="dpw" placeholder="Create a password for this device">
    </div>
    <div class="merr" id="derr"></div>
    <div class="mbtns">
      <button class="mbtn c" onclick="closeM('adm')">Cancel</button>
      <button class="mbtn p" id="addbtn" onclick="addDevice()">Add Device</button>
    </div>
  </div>
</div>

<script>
let cur = null, sid = null, corrCount = 0, pollT = null;

// ── TTS toggle ─────────────────────────────────────────────────────────────
let _ttsOn = true;
function toggleTTS(){
  fetch('/api/tts',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({enabled:!_ttsOn})})
  .then(r=>r.json()).then(d=>{ _ttsOn=d.enabled; _updateTTSBtn(); });
}
function _updateTTSBtn(){
  const btn=document.getElementById('tts-btn');
  if(!btn) return;
  btn.textContent=_ttsOn?'🔊 Voice ON':'🔇 Voice OFF';
  btn.className='hbtn '+(_ttsOn?'tts-on':'tts-off');
}
fetch('/api/tts').then(r=>r.json()).then(d=>{ _ttsOn=d.enabled; _updateTTSBtn(); });

let _speakQ=Promise.resolve();
function speakReply(text){
  if(!_ttsOn||!text) return;
  _speakQ=_speakQ.then(()=>
    fetch('/api/speak',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({text})})
    .then(r=>r.ok&&r.status!==204?r.blob():null)
    .then(blob=>{
      if(!blob||!blob.size) return;
      const url=URL.createObjectURL(blob);
      return new Promise(res=>{
        const a=new Audio(url);
        a.onended=()=>{URL.revokeObjectURL(url);res();};
        a.onerror=()=>{URL.revokeObjectURL(url);res();};
        a.play().catch(res);
      });
    }).catch(()=>{})
  );
}

// ── Modal ─────────────────────────────────────────────────────────────────
function openM(id){
  document.getElementById(id).classList.add('open');
  if(id==='sm'){
    fetch('/api/settings').then(r=>r.json()).then(d=>{
      document.getElementById('kinfo').textContent =
        d.set ? 'Current key: ' + d.partial : 'No key set.';
    });
  }
}
function closeM(id){ document.getElementById(id).classList.remove('open'); }
document.querySelectorAll('.backdrop').forEach(b =>
  b.addEventListener('click', e=>{ if(e.target===b) closeM(b.id); })
);

// ── Settings ──────────────────────────────────────────────────────────────
function saveSettings(){
  const k = document.getElementById('gk').value.trim();
  if(!k) return;
  fetch('/api/settings',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({gemini_key:k})
  }).then(()=>{ document.getElementById('gk').value=''; closeM('sm'); });
}

// ── Server IP (shown in header for ESP32 config) ──────────────────────────
fetch('/api/server-info').then(r=>r.json()).then(d=>{
  document.getElementById('srv-ip').textContent = 'Server: ' + d.ip + ':' + d.port;
});

// ── Devices ───────────────────────────────────────────────────────────────
function esc(s){ const d=document.createElement('div'); d.textContent=s; return d.innerHTML; }

function loadDevs(){
  fetch('/api/devices').then(r=>r.json()).then(devs=>{
    const el = document.getElementById('devlist');
    if(!devs.length){
      el.innerHTML = '<div class="no-dev">No devices added.<br>Click + Add Device<br>to get started.</div>';
      return;
    }
    el.innerHTML = devs.map(d=>`
      <div class="dev-item${cur===d.id?' active':''}" onclick="selDev('${d.id}','${esc(d.name)}')">
        <button class="dev-del" onclick="delDev(event,'${d.id}')">&#10005;</button>
        <div class="dev-name">
          <span class="dev-badge ${d.online?'on':'off'}"></span>${esc(d.name)}
        </div>
        <div class="dev-ip">${esc(d.ip)}</div>
      </div>`).join('');
  });
}

function delDev(e,mid){
  e.stopPropagation();
  if(!confirm('Remove this device?')) return;
  fetch('/api/devices/'+mid,{method:'DELETE'}).then(()=>{
    if(cur===mid){ cur=null; resetCam(); }
    loadDevs();
  });
}

function selDev(mid,name){
  cur=mid; loadDevs();
  document.getElementById('nocam').style.display='none';
  const img=document.getElementById('cam-img');
  img.src='/api/stream/'+mid;
  img.style.display='block';
  fetch('/api/chat/new',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({device_id:mid})
  }).then(r=>r.json()).then(d=>{
    sid=d.session_id; corrCount=0;
    document.getElementById('msgs').innerHTML=
      `<div class="hint">Camera connected: <strong>${name}</strong><br><br>`+
      `Tell me what pose you want to practice!<br>`+
      `Try: <em>"I want to do Padmasana"</em></div>`;
    startPoll();
    setStatus(false,null);
  });
}

function resetCam(){
  document.getElementById('nocam').style.display='flex';
  const img=document.getElementById('cam-img');
  img.style.display='none'; img.src='';
}

function addDevice(){
  const nm  = document.getElementById('dnm').value.trim();
  const mid = document.getElementById('dmid').value.trim();
  const pw  = document.getElementById('dpw').value.trim();
  const err = document.getElementById('derr');
  err.style.display='none';
  if(!mid||!pw){err.textContent='Machine ID and Password are required.';err.style.display='block';return;}
  const btn=document.getElementById('addbtn');
  btn.textContent='Connecting...'; btn.disabled=true;
  fetch('/api/devices/add',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({machine_id:mid,name:nm||mid,password:pw})
  }).then(r=>r.json()).then(d=>{
    btn.textContent='Add Device'; btn.disabled=false;
    if(d.error){err.textContent=d.error;err.style.display='block';}
    else{
      closeM('adm');
      ['dnm','dmid','dpw'].forEach(id=>document.getElementById(id).value='');
      loadDevs();
    }
  }).catch(e=>{btn.textContent='Add Device';btn.disabled=false;
    err.textContent='Error: '+e;err.style.display='block';
  });
}

// ── Chat ──────────────────────────────────────────────────────────────────
function handleKey(e){ if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();sendMsg();} }
function autoH(el){ el.style.height=''; el.style.height=Math.min(el.scrollHeight,90)+'px'; }

function sendMsg(){
  const ci=document.getElementById('ci');
  const txt=ci.value.trim();
  if(!txt||!sid) return;
  ci.value=''; ci.style.height='';
  addBub('u','chat',txt);
  fetch('/api/chat/'+sid+'/send',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({message:txt})
  }).then(r=>r.json()).then(d=>{
    if(d.reply){ addBub('a','chat',d.reply); speakReply(d.reply); }
    if(d.action==='start') setStatus(true,d.asana);
    else if(d.action==='stop') setStatus(false,null);
  }).catch(e=>addBub('a','chat','Error: '+e));
}

function addBub(role,type,content){
  const el=document.getElementById('msgs');
  const h=el.querySelector('.hint'); if(h) h.remove();
  const d=document.createElement('div');
  d.className='bub '+(role==='u'?'u':(type==='correction'?'c':'a'));
  if(role!=='u'){
    const lbl=document.createElement('div');
    lbl.className='bub-lbl';
    lbl.textContent=type==='correction'?'Pose Correction':'AI Coach';
    d.appendChild(lbl);
  }
  const t=document.createElement('div'); t.textContent=content;
  d.appendChild(t); el.appendChild(d);
  el.scrollTop=el.scrollHeight;
}

function setStatus(on,asana){
  document.getElementById('sdot').className='dot'+(on?' on':'');
  document.getElementById('stxt').textContent=on?'Monitoring Active':'Idle';
  document.getElementById('asana-box').style.display=on?'block':'none';
  if(asana) document.getElementById('asana-val').textContent=asana;
  document.getElementById('stopbtn').className='stopbtn'+(on?' show':'');
}

function stopMon(){
  if(!sid) return;
  fetch('/api/chat/'+sid+'/stop',{method:'POST'}).then(()=>{
    setStatus(false,null);
    addBub('a','chat','Monitoring stopped. Well done with your practice!');
  });
}

// ── Polling ───────────────────────────────────────────────────────────────
function startPoll(){ if(pollT) clearInterval(pollT); pollT=setInterval(poll,2500); }

function poll(){
  if(!sid) return;
  fetch('/api/chat/'+sid+'/status').then(r=>r.json()).then(s=>{
    setStatus(s.monitoring,s.asana);
  }).catch(()=>{});
  fetch('/api/chat/'+sid+'/messages').then(r=>r.json()).then(msgs=>{
    const corrs=msgs.filter(m=>m.type==='correction');
    if(corrs.length>corrCount){
      corrs.slice(corrCount).forEach(m=>{addBub('a','correction',m.content);speakReply(m.content);});
      corrCount=corrs.length;
    }
  }).catch(()=>{});
}

// ── Init ──────────────────────────────────────────────────────────────────
loadDevs();
setInterval(loadDevs, 15000);
</script>
</body>
</html>"""


@app.route("/")
def index():
    return Response(_HTML, content_type="text/html; charset=utf-8")


if __name__ == "__main__":
    try:
        ip = socket.gethostbyname(socket.gethostname())
    except Exception:
        ip = "127.0.0.1"
    print("=" * 60)
    print("  Yoga AI Coach  -  Web Application")
    print("=" * 60)
    print(f"  Open in browser : http://localhost:8080")
    print(f"  Also reachable  : http://{ip}:8080")
    print()
    print(f"  Gemini API key  : {'SET' if _gemini_key else 'NOT SET'}")
    print(f"  Devices         : {len(_devices)} registered")
    print()
    print(f"  ESP32 config    : Set SERVER_IP = \"{ip}\" in main.cpp")
    print(f"                    Set SERVER_PORT = 8080 in main.cpp")
    print("=" * 60)
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
