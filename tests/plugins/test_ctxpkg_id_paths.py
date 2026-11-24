"""Host-neutral path ids in the ``ctxpkg`` episode id extractor (structured-context step 2).

The extractor runs over rendered episode text that may come from any host, so a path id is any
absolute path of at least two segments, or a ``~/`` path — never one host's directory prefix.
A relative ``a/b/c``, a word-joined ``and/or`` and the path inside a URL are not ids.
"""

from __future__ import annotations

from plugins.context_engine.ctxpkg.episodes import extract_ids


def test_any_absolute_path_is_an_id():
    """An absolute path of two or more segments is an id on any host; a one-segment one is not."""
    assert extract_ids("/srv/proj/x/plan.md") == ["/srv/proj/x/plan.md"]
    assert extract_ids("/home/u/a.md") == ["/home/u/a.md"]
    assert extract_ids("/var/tmp/x/y") == ["/var/tmp/x/y"]
    # the rule is the segment count, not a known prefix: two segments are enough
    assert extract_ids("/a/b") == ["/a/b"]
    assert extract_ids("/tmp") == [], "a one-segment absolute path was kept as an id"
    # found inside a sentence, without eating the punctuation that ends it
    assert extract_ids("see (/srv/a/b.md), then stop") == ["/srv/a/b.md"]
    # first-seen order and dedupe are unchanged for neutral paths
    assert extract_ids("/srv/a/b /home/u/c.md /srv/a/b") == ["/srv/a/b", "/home/u/c.md"]


def test_home_tilde_path_is_an_id():
    """A ``~/`` path is one id, and its own tail is not re-read as an absolute path."""
    assert extract_ids("~/notes/a.md") == ["~/notes/a.md"]
    assert extract_ids("(see ~/x)") == ["~/x"]
    assert extract_ids("open ~/notes/a.md now") == ["~/notes/a.md"]
    # ``~user`` is not a home path, and the slash after it is word-joined
    assert extract_ids("~user/a/b") == []


def test_word_slash_and_url_paths_are_not_ids():
    """A slash after a word character, ``:`` or ``/`` joins text: no id starts there."""
    assert extract_ids("and/or") == []
    assert extract_ids("a/b/c") == [], "a relative path was kept as an id"
    assert extract_ids("https://x.org/a/b") == [], "the path inside a URL was kept as an id"
    assert extract_ids("file:/srv/a/b") == [], "a path after a scheme colon was kept as an id"
    assert extract_ids("//srv/a/b") == [], "a path after a doubled slash was kept as an id"


def test_trailing_dot_stripped():
    """A path ends before the punctuation that closes it; a dot inside the path stays."""
    assert extract_ids('"/srv/a/b.md."') == ["/srv/a/b.md"]
    assert extract_ids("/srv/a/b.md.") == ["/srv/a/b.md"]
    assert extract_ids("/srv/a/b.md, /srv/c/d.md; /srv/e/f.md:") == [
        "/srv/a/b.md", "/srv/c/d.md", "/srv/e/f.md",
    ]
    assert extract_ids("/srv/proj/v1.2/plan.md") == ["/srv/proj/v1.2/plan.md"]
