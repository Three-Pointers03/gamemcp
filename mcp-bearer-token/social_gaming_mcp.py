import asyncio
import os
import uuid
import json
from datetime import datetime, timedelta, timezone
from typing import Annotated, Optional, Literal
import logging
import functools
import inspect

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from fastmcp import FastMCP
from fastmcp.server.auth.providers.bearer import BearerAuthProvider, RSAKeyPair
from mcp.server.auth.provider import AccessToken
from mcp import ErrorData, McpError
from mcp.types import TextContent, INVALID_PARAMS, INTERNAL_ERROR
from supabase import create_client, Client


# ---------------- Env ----------------
load_dotenv()
TOKEN = os.environ.get("AUTH_TOKEN")
MY_NUMBER = os.environ.get("MY_NUMBER")
assert TOKEN, "Please set AUTH_TOKEN in your .env file"
assert MY_NUMBER is not None, "Please set MY_NUMBER in your .env file"

# Supabase configuration
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_ANON_KEY")
assert SUPABASE_URL, "Please set SUPABASE_URL in your .env file"
assert SUPABASE_KEY, "Please set SUPABASE_ANON_KEY in your .env file"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Logging configuration
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
ENABLE_TOOL_LOGS = os.environ.get("MCP_LOG_REQUESTS", "0").lower() in ("1", "true", "yes", "y")
LOG_FILE = os.environ.get("MCP_LOG_FILE")

logger = logging.getLogger("social_gaming_mcp")
if not logger.handlers:
    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    _stream_handler = logging.StreamHandler()
    _stream_handler.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    _stream_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(_stream_handler)
    if LOG_FILE:
        _file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
        _file_handler.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
        _file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(_file_handler)


# ---------------- Auth ----------------
class SimpleBearerAuthProvider(BearerAuthProvider):
    def __init__(self, token: str):
        k = RSAKeyPair.generate()
        super().__init__(public_key=k.public_key, jwks_uri=None, issuer=None, audience=None)
        self.token = token

    async def load_access_token(self, token: str) -> AccessToken | None:
        if token == self.token:
            return AccessToken(token=token, client_id="social-gaming-client", scopes=["*"], expires_at=None)
        return None


mcp = FastMCP(
    "GameParty MCP",
    auth=SimpleBearerAuthProvider(TOKEN),
)


# ---------------- Data Models ----------------
class LobbyChatMessage(BaseModel):
    timestamp: str
    user_id: str
    message: str


class LobbyState(BaseModel):
    lobby_id: str
    game_type: str
    players: list[str]
    max_players: int
    status: Literal["waiting", "ready", "active", "completed"]
    game_link: str
    created_at: str
    expires_at: str
    settings: dict | None = None


class QueueUser(BaseModel):
    user_id: str
    preferences: dict
    enqueued_at: str


class MatchmakingQueue(BaseModel):
    game_type: str
    waiting_users: list[QueueUser] = Field(default_factory=list)

    def add_user(self, user: QueueUser) -> None:
        self.waiting_users.append(user)

    def remove_user(self, user_id: str) -> None:
        self.waiting_users = [u for u in self.waiting_users if u.user_id != user_id]


# ---------------- In-memory State (removed; persisted via Supabase) ----------------

SUPPORTED_GAMES = {"skribbl", "deathbyai"}
DEFAULT_MATCH_SIZE = {"skribbl": 2, "deathbyai": 2}
DEFAULT_MAX_PLAYERS = {"skribbl": 8, "deathbyai": 4}
LOBBY_TTL_MINUTES = int(os.environ.get("LOBBY_TTL_MINUTES", "30"))


# ---------------- Helpers ----------------
def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _error(code, msg):
    raise McpError(ErrorData(code=code, message=msg))


def _ensure_queue(game_type: str) -> MatchmakingQueue:
    # Deprecated with Supabase-backed queues; kept for compatibility but unused.
    return MatchmakingQueue(game_type=game_type)


def _generate_skribbl_link(lobby_id: str) -> str:
    base = os.environ.get("SKRIBBL_LOBBY_BASE", "https://skribbl.io/")
    return f"{base}?{lobby_id[:8]}"


def _generate_deathbyai_link(lobby_id: str) -> str:
    base = os.environ.get("DEATHBYAI_LOBBY_BASE", "https://deathbyai.gg/")
    return f"{base}#r={lobby_id[:6]}"


def _generate_game_link(game_type: str, lobby_id: str) -> str:
    if game_type == "skribbl":
        return _generate_skribbl_link(lobby_id)
    if game_type == "deathbyai":
        return _generate_deathbyai_link(lobby_id)
    return f"https://example.com/{game_type}/{lobby_id}"


def _fetch_lobby_with_chat_dict(lobby_id: str) -> dict:
    lobby_resp = supabase.table("lobbies").select("*").eq("lobby_id", lobby_id).execute()
    if not lobby_resp.data:
        _error(INVALID_PARAMS, f"No lobby found for id {lobby_id}")
    lobby_row = lobby_resp.data[0]
    lobby = LobbyState(**{
        "lobby_id": lobby_row["lobby_id"],
        "game_type": lobby_row["game_type"],
        "players": lobby_row.get("players", []) or [],
        "max_players": lobby_row["max_players"],
        "status": lobby_row["status"],
        "game_link": lobby_row["game_link"],
        "created_at": str(lobby_row.get("created_at")),
        "expires_at": str(lobby_row.get("expires_at")),
        "settings": lobby_row.get("settings") or {},
    })
    chat_resp = (
        supabase.table("lobby_chat")
        .select("user_id,message,timestamp")
        .eq("lobby_id", lobby_id)
        .order("timestamp", desc=True)
        .limit(20)
        .execute()
    )
    messages = [
        LobbyChatMessage(timestamp=str(m.get("timestamp")), user_id=m.get("user_id"), message=m.get("message")).model_dump()
        for m in reversed(chat_resp.data or [])
    ]
    lobby_dict = lobby.model_dump()
    lobby_dict["chat"] = messages
    return lobby_dict


async def _join_queue_logic(
    gt: str,
    puch_user_id: str,
    preferences: Optional[str] = None,
) -> dict:
    if gt not in SUPPORTED_GAMES:
        _error(INVALID_PARAMS, f"Unsupported game_type '{gt}'. Supported: {sorted(SUPPORTED_GAMES)}")

    # If user already has a lobby, return that
    existing = supabase.table("user_lobbies").select("lobby_id").eq("user_id", puch_user_id).execute()
    if existing.data:
        lobby_id = existing.data[0]["lobby_id"]
        lobby_dict = _fetch_lobby_with_chat_dict(lobby_id)
        return {
            "status": "already_in_lobby",
            "lobby": lobby_dict,
        }

    # Prevent duplicate queue entries for the same game
    existing_q = (
        supabase.table("queue_users")
        .select("id", count="exact")
        .eq("game_type", gt)
        .eq("user_id", puch_user_id)
        .execute()
    )
    if existing_q.data:
        total_q = supabase.table("queue_users").select("id", count="exact").eq("game_type", gt).execute()
        waiting_count = max(0, (total_q.count or 0) - 1)
        return {
            "status": "already_in_queue",
            "game_type": gt,
            "waiting": waiting_count,
        }

    # Enqueue user
    pref_dict = {"raw": preferences} if preferences else {}
    supabase.table("queue_users").upsert({
        "game_type": gt,
        "user_id": puch_user_id,
        "preferences": pref_dict,
        "enqueued_at": _now(),
    }).execute()

    # Try to match immediately
    created = await _attempt_match(gt)
    if created:
        lobby = created[-1]
        lobby_dict = _fetch_lobby_with_chat_dict(lobby.lobby_id)
        return {
            "status": "match_found",
            "message": f"Match found! Lobby created for {gt}.",
            "lobby": lobby_dict,
        }
    else:
        total_q = supabase.table("queue_users").select("id", count="exact").eq("game_type", gt).execute()
        waiting_count = max(0, (total_q.count or 0) - 1)
        return {
            "status": "queued",
            "message": f"Added to {gt} queue.",
            "waiting": waiting_count,
        }


async def _create_lobby_internal(
    game_type: str,
    players: list[str],
    max_players: Optional[int] = None,
    settings: Optional[dict] = None,
) -> LobbyState:
    lobby_id = uuid.uuid4().hex
    created_at = datetime.now(timezone.utc)
    expires_at = created_at + timedelta(minutes=LOBBY_TTL_MINUTES)
    max_p = max_players or DEFAULT_MAX_PLAYERS.get(game_type, 4)
    link = _generate_game_link(game_type, lobby_id)
    status: Literal["waiting", "ready", "active", "completed"] = "ready" if len(players) >= 2 else "waiting"

    lobby_data = {
        "lobby_id": lobby_id,
        "game_type": game_type,
        "players": players,
        "max_players": max_p,
        "status": status,
        "game_link": link,
        "created_at": created_at.isoformat(),
        "expires_at": expires_at.isoformat(),
        "settings": settings or {},
    }
    supabase.table("lobbies").insert(lobby_data).execute()
    for uid in players:
        supabase.table("user_lobbies").upsert({"user_id": uid, "lobby_id": lobby_id}).execute()
    return LobbyState(**lobby_data)


async def _attempt_match(game_type: str) -> list[LobbyState]:
    """Try to form lobbies from the queue for a game. Returns newly created lobbies."""
    created: list[LobbyState] = []
    # Fetch queue ordered by enqueue time
    q_resp = (
        supabase.table("queue_users")
        .select("user_id,enqueued_at")
        .eq("game_type", game_type)
        .order("enqueued_at")
        .execute()
    )
    waiting = q_resp.data or []
    match_size = DEFAULT_MATCH_SIZE.get(game_type, 2)
    idx = 0
    while len(waiting) - idx >= match_size:
        batch = waiting[idx : idx + match_size]
        players = [u["user_id"] for u in batch]
        # Create lobby
        lobby = await _create_lobby_internal(game_type, players)
        created.append(lobby)
        # Remove matched users from queue
        supabase.table("queue_users").delete().eq("game_type", game_type).in_("user_id", players).execute()
        idx += match_size
    return created


async def _cleanup_task() -> None:
    while True:
        await asyncio.sleep(30)
        try:
            now = datetime.now(timezone.utc)
            # Delete completed lobbies
            supabase.table("lobbies").delete().eq("status", "completed").execute()
            # Delete expired lobbies
            supabase.table("lobbies").delete().lte("expires_at", now.isoformat()).execute()
            # Cleanup stale queue entries (> 30 minutes)
            cutoff = (now - timedelta(minutes=30)).isoformat()
            supabase.table("queue_users").delete().lt("enqueued_at", cutoff).execute()
            # Cleanup user->lobby mappings that point to non-existent lobbies
            active_resp = supabase.table("lobbies").select("lobby_id").execute()
            active_ids = {row.get("lobby_id") for row in (active_resp.data or [])}
            mappings = supabase.table("user_lobbies").select("user_id,lobby_id").execute()
            for mapping in (mappings.data or []):
                lobby_id = mapping.get("lobby_id")
                if lobby_id and lobby_id not in active_ids:
                    supabase.table("user_lobbies").delete().eq("user_id", mapping.get("user_id")).eq("lobby_id", lobby_id).execute()
        except Exception:
            # Do not crash background task
            pass


# ---------------- Rich Tool Descriptions ----------------
class RichToolDescription(BaseModel):
    description: str
    use_when: str
    side_effects: Optional[str] = None


def log_tool_io(tool_name: str):
    """Decorator to log tool inputs and outputs when MCP_LOG_REQUESTS is enabled.

    Logs a single line for request with tool name and arguments, and a single line for the response
    (truncated) to avoid overly large logs. Errors are logged at ERROR level with stack traces.
    """
    def decorator(func):
        if not inspect.iscoroutinefunction(func):
            return func

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            if ENABLE_TOOL_LOGS:
                try:
                    safe_kwargs = {k: (v if isinstance(v, (int, float, bool)) or v is None else str(v)) for k, v in kwargs.items()}
                    logger.info(f"MCP tool request: {tool_name} args={safe_kwargs}")
                except Exception:
                    logger.info(f"MCP tool request: {tool_name} (args logging failed)")
            try:
                result = await func(*args, **kwargs)
                if ENABLE_TOOL_LOGS:
                    # Avoid logging huge payloads
                    preview = None
                    try:
                        preview = json.dumps(result) if not isinstance(result, list) else (result[0].text[:500] + ("…" if len(result[0].text) > 500 else "") if result and hasattr(result[0], "text") else str(result)[:500])
                    except Exception:
                        preview = str(result)[:500]
                    logger.info(f"MCP tool response: {tool_name} -> {preview}")
                return result
            except Exception as e:
                logger.exception(f"MCP tool error: {tool_name}: {e}")
                raise

        return wrapper

    return decorator

JOIN_QUEUE_DESCRIPTION = RichToolDescription(
    description="Join matchmaking queue for a specified supported game (e.g., skribbl, deathbyai).",
    use_when="The user wants to find other players for a specific game and is willing to be matched.",
    side_effects="Adds the user to a persistent queue and may create a lobby if enough players are waiting.",
)

QUICK_MATCH_DESCRIPTION = RichToolDescription(
    description="Quick match the user into any available default game (skribbl).",
    use_when="The user wants to start playing right away without specifying a game.",
    side_effects="Attempts to match immediately in the default game's queue and may create a lobby.",
)

CREATE_GAME_LOBBY_DESCRIPTION = RichToolDescription(
    description="Create a new lobby for a supported game and return the shareable link.",
    use_when="A lobby is needed immediately or the user wants to host a room manually.",
    side_effects="Creates a persistent lobby with a generated link and TTL (30 minutes by default).",
)

GET_LOBBY_UPDATES_DESCRIPTION = RichToolDescription(
    description="Get the latest state for a lobby including players, status, link, and expiry.",
    use_when="The user is waiting for others to join or wants to check lobby status.",
)

LOBBY_CHAT_DESCRIPTION = RichToolDescription(
    description="Send a message to the lobby's waiting room chat.",
    use_when="Players want to coordinate before the game starts.",
    side_effects="Persists a chat message for the lobby.",
)


# ---------------- Tools ----------------
@mcp.tool
async def validate() -> str:
    return MY_NUMBER


@mcp.tool
async def about() -> dict:
    return {"name": mcp.name, "description": "An MCP server for meeting people to play games"}

@mcp.tool(description=JOIN_QUEUE_DESCRIPTION.model_dump_json())
@log_tool_io("join_queue")
async def join_queue(
    game_type: Annotated[str, Field(description="Supported game type: 'skribbl' or 'deathbyai'")],
    puch_user_id: Annotated[str, Field(description="Puch User Unique Identifier")],
    preferences: Annotated[Optional[str], Field(description="Free-form preferences such as 'casual', 'no mic', etc.")] = None,
) -> list[TextContent]:
    try:
        gt = game_type.strip().lower()
        result = await _join_queue_logic(gt, puch_user_id, preferences)
        return [TextContent(type="text", text=json.dumps(result))]
    except McpError:
        raise
    except Exception as e:
        _error(INTERNAL_ERROR, str(e))


@mcp.tool(description=QUICK_MATCH_DESCRIPTION.model_dump_json())
@log_tool_io("quick_match")
async def quick_match(
    puch_user_id: Annotated[str, Field(description="Puch User Unique Identifier")]
) -> list[TextContent]:
    try:
        # MVP: default to skribbl
        result = await _join_queue_logic("skribbl", puch_user_id, preferences="quickmatch")
        return [TextContent(type="text", text=json.dumps(result))]
    except McpError:
        raise
    except Exception as e:
        _error(INTERNAL_ERROR, str(e))


@mcp.tool(description=CREATE_GAME_LOBBY_DESCRIPTION.model_dump_json())
@log_tool_io("create_game_lobby")
async def create_game_lobby(
    game_type: Annotated[str, Field(description="Supported game type: 'skribbl' or 'deathbyai'")],
    max_players: Annotated[int, Field(description="Maximum players allowed in the lobby")] = 0,
    custom_settings: Annotated[Optional[str], Field(description="JSON string of custom settings")]=None,
    puch_user_id: Annotated[Optional[str], Field(description="Optional host user id; if present, will be added as the first player")]=None,
) -> list[TextContent]:
    try:
        gt = game_type.strip().lower()
        if gt not in SUPPORTED_GAMES:
            _error(INVALID_PARAMS, f"Unsupported game_type '{game_type}'. Supported: {sorted(SUPPORTED_GAMES)}")

        settings: dict = {}
        if custom_settings:
            try:
                settings = json.loads(custom_settings)
            except Exception:
                settings = {"raw": custom_settings}

        players: list[str] = [puch_user_id] if puch_user_id else []
        if puch_user_id:
            # remove from queue if present
            supabase.table("queue_users").delete().eq("game_type", gt).eq("user_id", puch_user_id).execute()

        lobby = await _create_lobby_internal(
            gt,
            players,
            max_players=max_players or DEFAULT_MAX_PLAYERS.get(gt, 4),
            settings=settings,
        )
        lobby_dict = _fetch_lobby_with_chat_dict(lobby.lobby_id)
        return [TextContent(type="text", text=json.dumps(lobby_dict))]
    except McpError:
        raise
    except Exception as e:
        _error(INTERNAL_ERROR, str(e))


@mcp.tool(description=GET_LOBBY_UPDATES_DESCRIPTION.model_dump_json())
@log_tool_io("get_lobby_updates")
async def get_lobby_updates(
    lobby_id: Annotated[str, Field(description="Lobby identifier")],
    puch_user_id: Annotated[Optional[str], Field(description="Optional Puch User ID; if provided and lobby_id is not found, will try to return the user's current lobby instead")] = None,
) -> list[TextContent]:
    try:
        lobby_dict = _fetch_lobby_with_chat_dict(lobby_id)
        return [TextContent(type="text", text=json.dumps(lobby_dict))]
    except McpError as e:
        not_found_payload = {
            "status": "not_found",
            "lobby_id": lobby_id,
            "message": "Lobby not found or may have expired. Use my_lobby to find your current lobby or create a new one.",
        }
        if puch_user_id:
            try:
                existing = supabase.table("user_lobbies").select("lobby_id").eq("user_id", puch_user_id).execute()
                if existing.data:
                    current_lobby_id = existing.data[0]["lobby_id"]
                    lobby_dict = _fetch_lobby_with_chat_dict(current_lobby_id)
                    return [TextContent(type="text", text=json.dumps({
                        "status": "redirected",
                        "from_lobby_id": lobby_id,
                        "lobby": lobby_dict,
                    }))]
            except Exception:
                # Fall back to not_found response if anything goes wrong during fallback
                pass
        return [TextContent(type="text", text=json.dumps(not_found_payload))]
    except Exception as e:
        _error(INTERNAL_ERROR, str(e))


@mcp.tool(description=LOBBY_CHAT_DESCRIPTION.model_dump_json())
@log_tool_io("lobby_chat")
async def lobby_chat(
    lobby_id: Annotated[str, Field(description="Lobby identifier")],
    puch_user_id: Annotated[str, Field(description="Puch User Unique Identifier")],
    message: Annotated[str, Field(description="Chat message to send")],
) -> list[TextContent]:
    try:
        if not message or not message.strip():
            _error(INVALID_PARAMS, "message cannot be empty")
        msg = message.strip()
        if len(msg) > 500:
            msg = msg[:500]

        # Ensure lobby exists
        lobby_resp = supabase.table("lobbies").select("lobby_id").eq("lobby_id", lobby_id).execute()
        if not lobby_resp.data:
            _error(INVALID_PARAMS, f"No lobby found for id {lobby_id}")
        # Persist chat
        supabase.table("lobby_chat").insert({
            "lobby_id": lobby_id,
            "user_id": puch_user_id,
            "message": msg,
            "timestamp": _now(),
        }).execute()
        # Return recent
        chat_resp = (
            supabase.table("lobby_chat")
            .select("user_id,message,timestamp")
            .eq("lobby_id", lobby_id)
            .order("timestamp", desc=True)
            .limit(20)
            .execute()
        )
        messages = [
            LobbyChatMessage(timestamp=str(m.get("timestamp")), user_id=m.get("user_id"), message=m.get("message")).model_dump()
            for m in reversed(chat_resp.data or [])
        ]
        return [TextContent(type="text", text=json.dumps({
            "ok": True,
            "lobby_id": lobby_id,
            "messages": messages,
        }))]
    except McpError:
        raise
    except Exception as e:
        _error(INTERNAL_ERROR, str(e))


# ---------------- Optional Utilities ----------------
@mcp.tool(description="Get the lobby for the current user if they are already matched")
@log_tool_io("my_lobby")
async def my_lobby(
    puch_user_id: Annotated[str, Field(description="Puch User Unique Identifier")],
) -> list[TextContent]:
    try:
        existing = supabase.table("user_lobbies").select("lobby_id").eq("user_id", puch_user_id).execute()
        if not existing.data:
            return [TextContent(type="text", text=json.dumps({"status": "no_lobby"}))]
        lobby_id = existing.data[0]["lobby_id"]
        lobby_dict = _fetch_lobby_with_chat_dict(lobby_id)
        return [TextContent(type="text", text=json.dumps(lobby_dict))]
    except Exception as e:
        _error(INTERNAL_ERROR, str(e))


@mcp.tool(description="Leave the matchmaking queue for a given game")
@log_tool_io("leave_queue")
async def leave_queue(
    game_type: Annotated[str, Field(description="Supported game type: 'skribbl' or 'deathbyai'")],
    puch_user_id: Annotated[str, Field(description="Puch User Unique Identifier")],
) -> list[TextContent]:
    try:
        gt = game_type.strip().lower()
        if gt not in SUPPORTED_GAMES:
            _error(INVALID_PARAMS, f"Unsupported game_type '{game_type}'.")
        supabase.table("queue_users").delete().eq("game_type", gt).eq("user_id", puch_user_id).execute()
        return [TextContent(type="text", text=json.dumps({"ok": True}))]
    except McpError:
        raise
    except Exception as e:
        _error(INTERNAL_ERROR, str(e))


@mcp.tool(description="Mark a lobby as active or completed")
@log_tool_io("set_lobby_status")
async def set_lobby_status(
    lobby_id: Annotated[str, Field(description="Lobby identifier")],
    status: Annotated[Literal["waiting", "ready", "active", "completed"], Field(description="New status")],
) -> list[TextContent]:
    try:
        # Update status in DB
        updates: dict = {"status": status}
        if status == "active":
            new_expiry = datetime.now(timezone.utc) + timedelta(minutes=LOBBY_TTL_MINUTES)
            updates["expires_at"] = new_expiry.isoformat()
        supabase.table("lobbies").update(updates).eq("lobby_id", lobby_id).execute()
        if status == "completed":
            # Free mapping immediately
            supabase.table("user_lobbies").delete().eq("lobby_id", lobby_id).execute()
        # Return latest lobby state with chat
        lobby_dict = _fetch_lobby_with_chat_dict(lobby_id)
        return [TextContent(type="text", text=json.dumps(lobby_dict))]
    except McpError:
        raise
    except Exception as e:
        _error(INTERNAL_ERROR, str(e))


# ---------------- Server ----------------
async def main():
    port = int(os.environ.get("PORT", "8087"))
    print(f"🎮 Starting Social Gaming MCP server on http://0.0.0.0:{port}  (Supabase-backed state)")
    # Background cleanup
    asyncio.create_task(_cleanup_task())
    await mcp.run_async("streamable-http", host="0.0.0.0", port=port)


if __name__ == "__main__":
    asyncio.run(main())


