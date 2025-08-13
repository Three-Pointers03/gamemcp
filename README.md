# Social Gaming MCP: Meet New People Through Games

This is a Model Context Protocol (MCP) server that turns Puch AI (a WhatsApp-based assistant) into your personal gaming matchmaker. The main motivation is to help you meet and connect with new people through fun, casual online games – no apps or websites needed, just chat with the AI!

## How It Works
- **Quick Matchmaking**: Message the AI to join queues for games like Skribbl.io (drawing/guessing) or Death by AI (survival challenges).
- **Private Lobbies**: Create shareable game rooms that auto-expire (default: 30 min).
- **Real-Time Updates**: Get notifications on who's joining and when the game starts.
- **Lobby Chat**: Talk with potential new friends while waiting.

Powered by Supabase for secure, user-scoped data storage. It's all about making social gaming effortless and helping you expand your circle through play!

## Quick Setup
1. Install dependencies: `uv venv && uv sync && source .venv/bin/activate`.
2. Set `.env` vars: `AUTH_TOKEN`, `MY_NUMBER`, `SUPABASE_URL`, `SUPABASE_ANON_KEY`.
3. Run: `cd mcp-bearer-token && python social_gaming_mcp.py`.


Connect in Puch AI: `/mcp connect DuPxpT549m

## Getting Help
- Puch AI Docs: https://puch.ai/mcp
- Discord: https://discord.gg/VMCnMvYx

Happy gaming and making friends! 🚀 #BuildWithPuch
