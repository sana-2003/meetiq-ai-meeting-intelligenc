"""
Meeting Intelligence System - FastAPI Backend
Real-time transcription + speaker diarization + action items + CRM push
Powered by Groq (free tier)
"""

import asyncio
import json
import os
import uuid
import tempfile
import wave
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# ── Whisper (optional)
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

# ── Groq (free AI) 
try:
    from groq import Groq
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    groq_client = Groq(api_key=GROQ_API_KEY)
    GROQ_AVAILABLE = True
except Exception as e:
    print(f"Groq not available: {e}")
    GROQ_AVAILABLE = False
    groq_client = None

# ── httpx (optional for CRM webhooks) 
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

# ── App 
app = FastAPI(title="MeetIQ API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── In-memory store 
sessions: dict[str, dict] = {}
connected_clients: dict[str, list[WebSocket]] = {}

# ── Speaker colors 
SPEAKER_COLORS = {
    "Speaker A": "#6366f1",
    "Speaker B": "#10b981",
    "Speaker C": "#f59e0b",
    "Speaker D": "#ef4444",
    "Speaker E": "#8b5cf6",
}

# ── Pydantic models 
class CRMPushRequest(BaseModel):
    session_id: str
    crm_type: str
    webhook_url: Optional[str] = None
    api_key: Optional[str] = None


# ── Speaker diarization (demo mode) 
def assign_speaker(segment_index: int, history: list) -> str:
    import random
    speaker_ids = list(SPEAKER_COLORS.keys())
    if not history:
        return speaker_ids[0]
    if random.random() < 0.3:
        current = list(set(history[-5:]))
        if len(current) < 3 and random.random() < 0.4:
            new_idx = min(len(set(history)), len(speaker_ids) - 1)
            return speaker_ids[new_idx]
        return random.choice(current)
    return history[-1]


# ── Transcription 
def transcribe_audio_chunk(audio_data: bytes) -> list[dict]:
    if WHISPER_AVAILABLE and len(audio_data) > 1000:
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                with wave.open(f.name, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(16000)
                    wf.writeframes(audio_data)
                model = whisper.load_model("base")
                result = model.transcribe(f.name, word_timestamps=True, language="en")
                os.unlink(f.name)
                return result.get("segments", [])
        except Exception as e:
            print(f"Whisper error: {e}")

    import time
    demo_segments = [
        "Let's kick off the Q3 product review. We have a lot to cover today.",
        "Revenue was up 23% but customer churn increased slightly in SMB segment.",
        "We need to reprioritize the onboarding redesign. Can you own that workstream?",
        "Yes, I'll have wireframes ready by Friday and share them in Figma.",
        "The API latency issues are blocking the enterprise tier launch. Needs urgent fix.",
        "I'll set up a war room with the infra team this week to get it unblocked.",
        "What's the status on mobile feature parity? Are we still on track?",
        "Targeting end of month but we need design sign-off first.",
        "Let's schedule a design review for Wednesday 2pm. I'll send the calendar invite.",
        "I need budget approval for the new analytics platform before we proceed.",
        "Send me the proposal and I'll escalate it to finance today.",
        "Any final blockers before we close out?",
        "All good from my side. This was very productive.",
        "Let's reconvene next Thursday for a progress check. Thanks everyone.",
    ]
    idx = int(time.time() * 1000) % len(demo_segments)
    return [{"text": demo_segments[idx], "start": 0.0, "end": 3.5, "no_speech_prob": 0.05}]


# ── AI: Action Items (Groq) 
async def generate_action_items(transcript: list[dict]) -> list[dict]:
    full_text = "\n".join([f"{s['speaker']}: {s['text']}" for s in transcript])

    if GROQ_AVAILABLE and groq_client:
        try:
            completion = groq_client.chat.completions.create(
                model="llama3-8b-8192",
                max_tokens=1000,
                messages=[{
                    "role": "user",
                    "content": f"""Extract action items from this meeting transcript. Return ONLY a valid JSON array, no explanation.

Format: [{{"text": "action description", "assignee": "person name or Team", "priority": "high/medium/low", "due_hint": "timeframe or TBD", "speaker_source": "speaker who committed"}}]

Transcript:
{full_text}

JSON array only:"""
                }]
            )
            raw = completion.choices[0].message.content.strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            # Find JSON array in response
            start = raw.find("[")
            end = raw.rfind("]") + 1
            if start != -1 and end > start:
                raw = raw[start:end]
            items = json.loads(raw)
            return [{"id": str(uuid.uuid4()), **item} for item in items]
        except Exception as e:
            print(f"Groq action items error: {e}")

    # Rule-based fallback
    action_keywords = ["will", "need to", "should", "by friday", "by end", "i'll", "we'll", "schedule", "send", "review", "assign", "escalate"]
    items = []
    for seg in transcript:
        if any(kw in seg["text"].lower() for kw in action_keywords):
            priority = "high" if any(w in seg["text"].lower() for w in ["urgent", "blocking", "critical", "asap"]) else "medium"
            items.append({
                "id": str(uuid.uuid4()),
                "text": seg["text"],
                "assignee": seg["speaker"],
                "priority": priority,
                "due_hint": "End of sprint",
                "speaker_source": seg["speaker"]
            })
    return items[:10]


# ── AI: Summary (Groq) 
async def generate_summary(transcript: list[dict]) -> str:
    full_text = "\n".join([f"{s['speaker']}: {s['text']}" for s in transcript])

    if GROQ_AVAILABLE and groq_client:
        try:
            completion = groq_client.chat.completions.create(
                model="llama3-8b-8192",
                max_tokens=400,
                messages=[{
                    "role": "user",
                    "content": f"""Summarize this meeting in 3 concise sentences. Cover: main topics, key decisions, and outcomes. Write directly, no preamble.

Transcript:
{full_text}

Summary:"""
                }]
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            print(f"Groq summary error: {e}")

    # Fallback
    speakers = list(set(s["speaker"] for s in transcript))
    return (f"Meeting with {len(speakers)} participant(s) covered {len(transcript)} discussion points. "
            f"Key topics included project updates, blockers, and team assignments. "
            f"Action items were identified and assigned to respective team members.")


# ── CRM Push 
async def push_to_crm(session: dict, config: CRMPushRequest) -> dict:
    payload = {
        "meeting_title": session["title"],
        "date": session["created_at"],
        "duration_minutes": round(session["duration"] / 60, 1),
        "participants": session["participants"],
        "summary": session["summary"],
        "action_items": session["action_items"],
    }

    if config.crm_type == "slack":
        slack_msg = {
            "text": f"📋 *{session['title']}*",
            "blocks": [
                {"type": "header", "text": {"type": "plain_text", "text": f"📋 {session['title']}"}},
                {"type": "section", "text": {"type": "mrkdwn", "text": f"*Summary:*\n{session['summary']}"}},
                {"type": "section", "text": {"type": "mrkdwn", "text": f"*Participants:* {', '.join(session['participants'])}"}},
                {"type": "section", "text": {"type": "mrkdwn", "text": "*Action Items:*\n" + "\n".join([f"• {a['text']} → {a['assignee']} [{a['priority']}]" for a in session['action_items'][:5]])}},
            ]
        }
        if config.webhook_url and HTTPX_AVAILABLE:
            async with httpx.AsyncClient() as client:
                resp = await client.post(config.webhook_url, json=slack_msg, timeout=10)
                return {"status": "sent", "platform": "slack", "http_status": resp.status_code}
        return {"status": "simulated", "platform": "slack", "payload": slack_msg}

    elif config.crm_type == "hubspot":
        return {"status": "simulated", "platform": "hubspot", "payload": payload}

    elif config.crm_type == "webhook":
        if config.webhook_url and HTTPX_AVAILABLE:
            async with httpx.AsyncClient() as client:
                resp = await client.post(config.webhook_url, json=payload, timeout=10)
                return {"status": "sent", "platform": "webhook", "http_status": resp.status_code}
        return {"status": "simulated", "platform": "webhook", "payload": payload}

    return {"status": "simulated", "platform": config.crm_type, "payload": payload}


# ── Broadcast 
async def broadcast(session_id: str, event: str, data: dict):
    dead = []
    for ws in connected_clients.get(session_id, []):
        try:
            await ws.send_json({"event": event, "data": data})
        except Exception:
            dead.append(ws)
    for ws in dead:
        connected_clients[session_id].remove(ws)


# ── Finalize session 
async def finalize_session(session_id: str):
    session = sessions[session_id]
    session["status"] = "processing"
    session["duration"] = datetime.now().timestamp() - session["_start_time"]

    await broadcast(session_id, "status", {"status": "processing", "message": "Extracting action items with Groq AI..."})
    session["action_items"] = await generate_action_items(session["transcript"])
    await broadcast(session_id, "action_items", {"items": session["action_items"]})

    await broadcast(session_id, "status", {"status": "processing", "message": "Generating summary..."})
    session["summary"] = await generate_summary(session["transcript"])
    await broadcast(session_id, "summary", {"text": session["summary"]})

    session["status"] = "completed"
    await broadcast(session_id, "status", {"status": "completed", "message": "Meeting intelligence ready ✓"})


# ── REST Endpoints 
@app.post("/api/sessions")
async def create_session(title: str = "Team Meeting"):
    sid = str(uuid.uuid4())
    sessions[sid] = {
        "session_id": sid,
        "title": title,
        "created_at": datetime.now().isoformat(),
        "status": "live",
        "participants": [],
        "transcript": [],
        "action_items": [],
        "summary": "",
        "duration": 0.0,
        "_start_time": datetime.now().timestamp(),
        "_speaker_history": [],
    }
    connected_clients[sid] = []
    return {"session_id": sid, "message": "Session created"}


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    if session_id not in sessions:
        raise HTTPException(404, "Session not found")
    s = {k: v for k, v in sessions[session_id].items() if not k.startswith("_")}
    return s


@app.get("/api/sessions")
async def list_sessions():
    return [{k: v for k, v in s.items() if not k.startswith("_")} for s in sessions.values()]


@app.post("/api/sessions/{session_id}/finalize")
async def finalize_endpoint(session_id: str):
    if session_id not in sessions:
        raise HTTPException(404, "Session not found")
    await finalize_session(session_id)
    return {"message": "Finalized"}


@app.post("/api/sessions/{session_id}/upload-audio")
async def upload_audio(session_id: str, background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if session_id not in sessions:
        raise HTTPException(404, "Session not found")
    audio_data = await file.read()
    background_tasks.add_task(process_audio_bg, session_id, audio_data)
    return {"message": "Processing started", "bytes": len(audio_data)}


async def process_audio_bg(session_id: str, audio_data: bytes):
    session = sessions[session_id]
    await broadcast(session_id, "status", {"status": "processing", "message": "Transcribing audio..."})
    segments = transcribe_audio_chunk(audio_data)
    for i, seg in enumerate(segments):
        speaker = assign_speaker(i, session["_speaker_history"])
        session["_speaker_history"].append(speaker)
        text = seg.get("text", "").strip()
        if text:
            item = {
                "id": str(uuid.uuid4()),
                "speaker": speaker,
                "text": text,
                "start_time": seg.get("start", i * 3.0),
                "end_time": seg.get("end", i * 3.0 + 3.0),
                "confidence": round(1 - seg.get("no_speech_prob", 0.05), 3),
            }
            session["transcript"].append(item)
            if speaker not in session["participants"]:
                session["participants"].append(speaker)
            await broadcast(session_id, "transcript_segment", item)
    await finalize_session(session_id)


@app.post("/api/sessions/{session_id}/crm-push")
async def crm_push(session_id: str, config: CRMPushRequest):
    if session_id not in sessions:
        raise HTTPException(404, "Session not found")
    return await push_to_crm(sessions[session_id], config)


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    sessions.pop(session_id, None)
    connected_clients.pop(session_id, None)
    return {"message": "Deleted"}


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "whisper_available": WHISPER_AVAILABLE,
        "groq_available": GROQ_AVAILABLE,
        "mode": "production" if WHISPER_AVAILABLE else "demo",
        "ai": "groq/llama3-8b" if GROQ_AVAILABLE else "rule-based"
    }


# ── WebSocket 
@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    if session_id not in sessions:
        await websocket.send_json({"event": "error", "data": {"message": "Session not found"}})
        await websocket.close()
        return

    connected_clients.setdefault(session_id, []).append(websocket)
    session = sessions[session_id]
    segment_counter = 0

    try:
        await websocket.send_json({"event": "connected", "data": {"session_id": session_id}})
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive(), timeout=60.0)
            except asyncio.TimeoutError:
                await websocket.send_json({"event": "ping", "data": {}})
                continue

            if data["type"] == "websocket.disconnect":
                break

            if data["type"] == "websocket.receive":
                if data.get("bytes"):
                    # Real PCM audio chunk
                    audio_buffer = data["bytes"]
                    if len(audio_buffer) >= 48000:
                        segs = transcribe_audio_chunk(audio_buffer)
                        for seg in segs:
                            text = seg.get("text", "").strip()
                            if text and seg.get("no_speech_prob", 1.0) < 0.5:
                                speaker = assign_speaker(segment_counter, session["_speaker_history"])
                                session["_speaker_history"].append(speaker)
                                item = {
                                    "id": str(uuid.uuid4()),
                                    "speaker": speaker,
                                    "text": text,
                                    "start_time": seg.get("start", segment_counter * 3.0),
                                    "end_time": seg.get("end", segment_counter * 3.0 + 3.0),
                                    "confidence": round(1 - seg.get("no_speech_prob", 0.05), 3),
                                }
                                session["transcript"].append(item)
                                if speaker not in session["participants"]:
                                    session["participants"].append(speaker)
                                segment_counter += 1
                                await broadcast(session_id, "transcript_segment", item)

                elif data.get("text"):
                    msg = json.loads(data["text"])
                    if msg.get("type") == "demo_segment":
                        speaker = assign_speaker(segment_counter, session["_speaker_history"])
                        session["_speaker_history"].append(speaker)
                        item = {
                            "id": str(uuid.uuid4()),
                            "speaker": speaker,
                            "text": msg["text"],
                            "start_time": segment_counter * 3.5,
                            "end_time": segment_counter * 3.5 + 3.5,
                            "confidence": 0.97,
                        }
                        session["transcript"].append(item)
                        if speaker not in session["participants"]:
                            session["participants"].append(speaker)
                        segment_counter += 1
                        await broadcast(session_id, "transcript_segment", item)
                    elif msg.get("type") == "finalize":
                        await finalize_session(session_id)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WS error: {e}")
    finally:
        clients = connected_clients.get(session_id, [])
        if websocket in clients:
            clients.remove(websocket)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
