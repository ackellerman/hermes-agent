"""Category resolution on REAL discord.py objects (fork addition to the #61593 carry).

The carried PR tests use hand-written doubles. These build ``Guild`` / ``CategoryChannel`` /
``TextChannel`` / ``ForumChannel`` / ``VoiceChannel`` / ``Thread`` offline from gateway payloads on a
mock connection state, so ``Thread.parent`` and ``.category`` resolve through the real guild cache
exactly as they do for a live message. All IDs and names are fake.
"""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import discord
import pytest

from plugins.platforms.discord.adapter import DiscordAdapter


def _real_discord():
    real = sys.modules.get("discord")
    if real is None or not hasattr(real, "__file__") or not hasattr(real, "Guild"):
        pytest.skip("real discord.py is not installed")
    return real


@pytest.fixture
def real_guild():
    d = _real_discord()
    state = MagicMock()
    state.self_id = 1
    state.member_cache_flags = d.MemberCacheFlags.none()
    guild = d.Guild(
        data={"id": "1", "name": "g", "roles": [], "emojis": [], "stickers": [],
              "features": [], "channels": [], "threads": []},
        state=state,
    )

    def channel(cls, data):
        payload = {"guild_id": "1", "position": 0, "permission_overwrites": [], "nsfw": False}
        payload.update(data)
        obj = cls(state=state, guild=guild, data=payload)
        guild._add_channel(obj)
        return obj

    def thread(thread_id, parent_id):
        return d.Thread(guild=guild, state=state, data={
            "id": thread_id, "guild_id": "1", "parent_id": parent_id, "owner_id": "9",
            "name": "t", "type": 11, "message_count": 0, "member_count": 0,
            "rate_limit_per_user": 0,
            "thread_metadata": {"archived": False, "auto_archive_duration": 1440,
                                "archive_timestamp": "2026-01-01T00:00:00+00:00", "locked": False},
        })

    shapes = SimpleNamespace(state=state, guild=guild, thread=thread)
    shapes.category = channel(d.CategoryChannel, {"id": "500", "type": 4, "name": "Projects"})
    shapes.text = channel(d.TextChannel, {"id": "600", "type": 0, "name": "general", "parent_id": "500"})
    shapes.forum = channel(d.ForumChannel, {"id": "700", "type": 15, "name": "app-forum",
                                            "parent_id": "500", "available_tags": [], "flags": 0})
    shapes.loose_forum = channel(d.ForumChannel, {"id": "710", "type": 15, "name": "loose",
                                                  "available_tags": [], "flags": 0})
    shapes.voice = channel(d.VoiceChannel, {"id": "720", "type": 2, "name": "vc", "parent_id": "500",
                                            "bitrate": 64000, "user_limit": 0})
    return shapes


def _category_keys(channel):
    adapter = object.__new__(DiscordAdapter)
    return adapter._discord_category_keys_from_channel(channel)


CATEGORY_KEYS = {"500", "Projects", "#Projects"}


def test_real_text_channel_in_category(real_guild):
    assert _category_keys(real_guild.text) == CATEGORY_KEYS


def test_real_thread_in_text_channel_inherits_parent_category(real_guild):
    assert _category_keys(real_guild.thread("801", "600")) == CATEGORY_KEYS


def test_real_forum_post_inherits_forum_category(real_guild):
    assert _category_keys(real_guild.thread("802", "700")) == CATEGORY_KEYS


def test_real_forum_without_category_yields_nothing(real_guild):
    assert _category_keys(real_guild.thread("803", "710")) == set()


def test_real_thread_with_uncached_parent_yields_nothing_without_raising(real_guild):
    d = _real_discord()
    orphan = real_guild.thread("804", "999")
    # The raw property raises; the key builder must swallow it (no fetch, no exception).
    with pytest.raises(d.ClientException):
        orphan.category_id
    assert _category_keys(orphan) == set()


def test_real_partial_messageable_uncached_thread_yields_nothing(real_guild):
    d = _real_discord()
    assert _category_keys(d.PartialMessageable(state=real_guild.state, id=805, guild_id=1)) == set()


def test_real_voice_text_chat_in_category_matches(real_guild):
    # Documented scope: every channel type under a listed category is free-response.
    assert _category_keys(real_guild.voice) == CATEGORY_KEYS


def test_real_shapes_make_no_fetch_call(real_guild):
    for channel in (real_guild.text, real_guild.thread("806", "700"), real_guild.thread("807", "999")):
        _category_keys(channel)
    assert [c[0] for c in real_guild.state.method_calls if "fetch" in c[0] or "http" in c[0]] == []
