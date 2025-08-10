import asyncio
import os
import uuid
import json
from datetime import datetime, timedelta, timezone
from typing import Annotated, Optional, Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from fastmcp import FastMCP
from fastmcp.server.auth.providers.bearer import BearerAuthProvider, RSAKeyPair
from mcp.server.auth.provider import AccessToken
from mcp import ErrorData, McpError
from mcp.types import TextContent, INVALID_PARAMS, INTERNAL_ERROR


# ---------------- Env ----------------
load_dotenv()
TOKEN = os.environ.get("AUTH_TOKEN")
MY_NUMBER = os.environ.get("MY_NUMBER")
assert TOKEN, "Please set AUTH_TOKEN in your .env file"
assert MY_NUMBER is not None, "Please set MY_NUMBER in your .env file"


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
    "Social Gaming MCP Server",
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
    chat: list[LobbyChatMessage] = Field(default_factory=list)


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


# ---------------- In-memory State ----------------
STATE_LOCK = asyncio.Lock()
QUEUES: dict[str, MatchmakingQueue] = {}
LOBBIES: dict[str, LobbyState] = {}
USER_TO_LOBBY: dict[str, str] = {}
USER_IN_QUEUES: dict[str, set[str]] = {}

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
    if game_type not in QUEUES:
        QUEUES[game_type] = MatchmakingQueue(game_type=game_type)
    return QUEUES[game_type]


def _generate_skribbl_link(lobby_id: str) -> str:
    base = os.environ.get("SKRIBBL_LOBBY_BASE", "https://skribbl.io/")
    return f"{base}?private={lobby_id[:6]}"


def _generate_deathbyai_link(lobby_id: str) -> str:
    base = os.environ.get("DEATHBYAI_LOBBY_BASE", "https://deathbyai.gg/")
    return f"{base}#r={lobby_id[:6]}"


def _generate_game_link(game_type: str, lobby_id: str) -> str:
    if game_type == "skribbl":
        return _generate_skribbl_link(lobby_id)
    if game_type == "deathbyai":
        return _generate_deathbyai_link(lobby_id)
    return f"https://example.com/{game_type}/{lobby_id}"


async def _join_queue_logic(
    gt: str,
    puch_user_id: str,
    preferences: Optional[str] = None,
) -> dict:
    if gt not in SUPPORTED_GAMES:
        _error(INVALID_PARAMS, f"Unsupported game_type '{gt}'. Supported: {sorted(SUPPORTED_GAMES)}")

    async with STATE_LOCK:
        # If user already has a lobby, return that
        existing_lobby_id = USER_TO_LOBBY.get(puch_user_id)
        if existing_lobby_id and existing_lobby_id in LOBBIES:
            lobby = LOBBIES[existing_lobby_id]
            return {
                "status": "already_in_lobby",
                "lobby": lobby.model_dump(),
            }

        # Prevent duplicate queue entries for the same game
        user_games = USER_IN_QUEUES.setdefault(puch_user_id, set())
        if gt in user_games:
            queue = _ensure_queue(gt)
            waiting_count = max(0, len(queue.waiting_users) - 1)
            return {
                "status": "already_in_queue",
                "game_type": gt,
                "waiting": waiting_count,
            }

        queue = _ensure_queue(gt)
        user_games.add(gt)
        pref_dict = {"raw": preferences} if preferences else {}
        queue.add_user(QueueUser(user_id=puch_user_id, preferences=pref_dict, enqueued_at=_now()))

        # Try to match immediately
        created = await _attempt_match(gt)
        if created:
            lobby = created[-1]
            return {
                "status": "match_found",
                "message": f"Match found! Lobby created for {gt}.",
                "lobby": lobby.model_dump(),
            }
        else:
            return {
                "status": "queued",
                "message": f"Added to {gt} queue.",
                "waiting": len(queue.waiting_users) - 1,
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

    lobby = LobbyState(
        lobby_id=lobby_id,
        game_type=game_type,
        players=players,
        max_players=max_p,
        status=status,
        game_link=link,
        created_at=created_at.isoformat(),
        expires_at=expires_at.isoformat(),
        settings=settings or {},
    )
    LOBBIES[lobby_id] = lobby
    for uid in players:
        USER_TO_LOBBY[uid] = lobby_id
    return lobby


async def _attempt_match(game_type: str) -> list[LobbyState]:
    """Try to form lobbies from the queue for a game. Returns newly created lobbies."""
    created: list[LobbyState] = []
    queue = _ensure_queue(game_type)
    match_size = DEFAULT_MATCH_SIZE.get(game_type, 2)
    while len(queue.waiting_users) >= match_size:
        batch = queue.waiting_users[:match_size]
        queue.waiting_users = queue.waiting_users[match_size:]
        players = [u.user_id for u in batch]
        for uid in players:
            USER_IN_QUEUES.get(uid, set()).discard(game_type)
        lobby = await _create_lobby_internal(game_type, players)
        created.append(lobby)
    return created


async def _cleanup_task() -> None:
    while True:
        await asyncio.sleep(30)
        try:
            async with STATE_LOCK:
                now = datetime.now(timezone.utc)
                # Cleanup lobbies
                expired_lobbies = [
                    l_id
                    for l_id, lobby in list(LOBBIES.items())
                    if datetime.fromisoformat(lobby.expires_at) <= now
                    or lobby.status == "completed"
                ]
                for l_id in expired_lobbies:
                    lobby = LOBBIES.pop(l_id, None)
                    if lobby:
                        for uid in list(lobby.players):
                            USER_TO_LOBBY.pop(uid, None)

                # Cleanup stale queue entries (> 30 minutes)
                for game_type, queue in QUEUES.items():
                    fresh: list[QueueUser] = []
                    for u in queue.waiting_users:
                        t = datetime.fromisoformat(u.enqueued_at)
                        if (now - t) <= timedelta(minutes=30):
                            fresh.append(u)
                        else:
                            USER_IN_QUEUES.get(u.user_id, set()).discard(game_type)
                    queue.waiting_users = fresh
        except Exception:
            # Do not crash background task
            pass


# ---------------- Rich Tool Descriptions ----------------
class RichToolDescription(BaseModel):
    description: str
    use_when: str
    side_effects: Optional[str] = None


JOIN_QUEUE_DESCRIPTION = RichToolDescription(
    description="Join matchmaking queue for a specified supported game (e.g., skribbl, deathbyai).",
    use_when="The user wants to find other players for a specific game and is willing to be matched.",
    side_effects="Adds the user to an in-memory queue and may create a lobby if enough players are waiting.",
)

QUICK_MATCH_DESCRIPTION = RichToolDescription(
    description="Quick match the user into any available default game (skribbl).",
    use_when="The user wants to start playing right away without specifying a game.",
    side_effects="Attempts to match immediately in the default game's queue and may create a lobby.",
)

CREATE_GAME_LOBBY_DESCRIPTION = RichToolDescription(
    description="Create a new lobby for a supported game and return the shareable link.",
    use_when="A lobby is needed immediately or the user wants to host a room manually.",
    side_effects="Creates an in-memory lobby with a generated link and TTL (30 minutes by default).",
)

GET_LOBBY_UPDATES_DESCRIPTION = RichToolDescription(
    description="Get the latest state for a lobby including players, status, link, and expiry.",
    use_when="The user is waiting for others to join or wants to check lobby status.",
)

LOBBY_CHAT_DESCRIPTION = RichToolDescription(
    description="Send a message to the lobby's waiting room chat.",
    use_when="Players want to coordinate before the game starts.",
    side_effects="Appends a chat message in the lobby. Keeps only in-memory chat for the demo.",
)


# ---------------- Tools ----------------
@mcp.tool
async def validate() -> str:
    return MY_NUMBER


@mcp.tool(description=JOIN_QUEUE_DESCRIPTION.model_dump_json())
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

        async with STATE_LOCK:
            players: list[str] = [puch_user_id] if puch_user_id else []
            if puch_user_id:
                # remove from queue if present
                if puch_user_id in USER_IN_QUEUES:
                    if gt in USER_IN_QUEUES[puch_user_id]:
                        _ensure_queue(gt).remove_user(puch_user_id)
                        USER_IN_QUEUES[puch_user_id].discard(gt)

            lobby = await _create_lobby_internal(
                gt,
                players,
                max_players=max_players or DEFAULT_MAX_PLAYERS.get(gt, 4),
                settings=settings,
            )
            return [TextContent(type="text", text=json.dumps(lobby.model_dump()))]
    except McpError:
        raise
    except Exception as e:
        _error(INTERNAL_ERROR, str(e))


@mcp.tool(description=GET_LOBBY_UPDATES_DESCRIPTION.model_dump_json())
async def get_lobby_updates(
    lobby_id: Annotated[str, Field(description="Lobby identifier")]
) -> list[TextContent]:
    try:
        async with STATE_LOCK:
            lobby = LOBBIES.get(lobby_id)
            if not lobby:
                _error(INVALID_PARAMS, f"No lobby found for id {lobby_id}")
            return [TextContent(type="text", text=json.dumps(lobby.model_dump()))]
    except McpError:
        raise
    except Exception as e:
        _error(INTERNAL_ERROR, str(e))


@mcp.tool(description=LOBBY_CHAT_DESCRIPTION.model_dump_json())
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

        async with STATE_LOCK:
            lobby = LOBBIES.get(lobby_id)
            if not lobby:
                _error(INVALID_PARAMS, f"No lobby found for id {lobby_id}")
            # Allow chat even if user not in players for coordination
            lobby.chat.append(LobbyChatMessage(timestamp=_now(), user_id=puch_user_id, message=msg))
            return [TextContent(type="text", text=json.dumps({
                "ok": True,
                "lobby_id": lobby_id,
                "messages": [m.model_dump() for m in lobby.chat[-20:]],
            }))]
    except McpError:
        raise
    except Exception as e:
        _error(INTERNAL_ERROR, str(e))


# ---------------- Optional Utilities ----------------
@mcp.tool(description="Get the lobby for the current user if they are already matched")
async def my_lobby(
    puch_user_id: Annotated[str, Field(description="Puch User Unique Identifier")],
) -> list[TextContent]:
    try:
        async with STATE_LOCK:
            lobby_id = USER_TO_LOBBY.get(puch_user_id)
            if not lobby_id or lobby_id not in LOBBIES:
                return [TextContent(type="text", text=json.dumps({"status": "no_lobby"}))]
            return [TextContent(type="text", text=json.dumps(LOBBIES[lobby_id].model_dump()))]
    except Exception as e:
        _error(INTERNAL_ERROR, str(e))


@mcp.tool(description="Leave the matchmaking queue for a given game")
async def leave_queue(
    game_type: Annotated[str, Field(description="Supported game type: 'skribbl' or 'deathbyai'")],
    puch_user_id: Annotated[str, Field(description="Puch User Unique Identifier")],
) -> list[TextContent]:
    try:
        gt = game_type.strip().lower()
        async with STATE_LOCK:
            if gt not in SUPPORTED_GAMES:
                _error(INVALID_PARAMS, f"Unsupported game_type '{game_type}'.")
            _ensure_queue(gt).remove_user(puch_user_id)
            USER_IN_QUEUES.get(puch_user_id, set()).discard(gt)
            return [TextContent(type="text", text=json.dumps({"ok": True}))]
    except McpError:
        raise
    except Exception as e:
        _error(INTERNAL_ERROR, str(e))


@mcp.tool(description="Mark a lobby as active or completed")
async def set_lobby_status(
    lobby_id: Annotated[str, Field(description="Lobby identifier")],
    status: Annotated[Literal["waiting", "ready", "active", "completed"], Field(description="New status")],
) -> list[TextContent]:
    try:
        async with STATE_LOCK:
            lobby = LOBBIES.get(lobby_id)
            if not lobby:
                _error(INVALID_PARAMS, f"No lobby found for id {lobby_id}")
            lobby.status = status
            # extend TTL when activated
            if status == "active":
                new_expiry = datetime.now(timezone.utc) + timedelta(minutes=LOBBY_TTL_MINUTES)
                lobby.expires_at = new_expiry.isoformat()
            if status == "completed":
                # Upon completion, users can rematch; free mapping immediately
                for uid in list(lobby.players):
                    USER_TO_LOBBY.pop(uid, None)
            return [TextContent(type="text", text=json.dumps(lobby.model_dump()))]
    except McpError:
        raise
    except Exception as e:
        _error(INTERNAL_ERROR, str(e))


# ---------------- Server ----------------
async def main():
    port = int(os.environ.get("PORT", "8087"))
    print(f"🎮 Starting Social Gaming MCP server on http://0.0.0.0:{port}  (in-memory state)")
    # Background cleanup
    asyncio.create_task(_cleanup_task())
    await mcp.run_async("streamable-http", host="0.0.0.0", port=port)


if __name__ == "__main__":
    asyncio.run(main())


