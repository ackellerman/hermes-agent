"""``session.create`` reports, on the canonical wire, whether *this* call made the row.

The reply is a ``CanonicalSnapshot`` and ``info`` is a ``CanonicalSnapshotInfo`` — ``extra="forbid"``,
so ``info['created']`` is part of the declared contract the owner validates its own replies against
(``tui_gateway/contracts/canonical_sessions.py``; ``gateway/session_controls.py`` ``dispatch`` →
``check_result``). These cases drive the real ``AuthorityConnection.dispatch``: its params/result
contract checks, its ``_create``, and the real ``stage_local_session`` / ``commit_local_session`` /
``restore_local_session`` beneath it. A key added to ``info`` without the declared field therefore
fails here rather than in a full gateway run.

The three meanings pinned: a titled create that mints says true and one that resolves says false; an
untitled create (the ordinary ``session.create`` behind every plain CLI start) always mints, so true;
an exact retry of a ``request_id`` whose earlier attempt committed reports false, because
``stage_local_session`` hands back the row itself rather than a staged receipt
(``gateway/session_local.py``), and this request_id's row is already there.
"""
import json

import pytest


def _owner_home(monkeypatch):
    """A real owner on a real config: ``_load_gateway_config`` is not patched. ``run._hermes_home`` is
    rebound because another gateway module may have imported ``gateway.run`` under a different home."""
    from gateway import run
    from hermes_constants import get_hermes_home
    from tests.gateway.test_session_follow_profile_config import _owner
    home = get_hermes_home()
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(run, '_hermes_home', home)
    (home / 'config.yaml').write_text(json.dumps({'model': {
        'provider': 'custom', 'base_url': 'http://127.0.0.1:1/v1', 'default': 'wire-model',
        'api_key': 'wire-key'}}))
    store, runner, _authority, owner = _owner(home)
    runner._peek_session_state = lambda key: None
    return store, owner, home


async def _create(owner, home, request_id, **extra):
    return await owner.dispatch({'id': request_id, 'method': 'session.create', 'params': dict(
        request_id=request_id, source='cli', cwd=str(home), toolsets=[],
        provider='custom', model='wire-model', **extra)})


@pytest.mark.asyncio
async def test_a_canonical_titled_create_reports_the_row_it_made(monkeypatch):
    """Resolve-or-create over the wire: a fresh title reports ``created: true``, the same title again
    an existing row (``false``, the id it resolved), and both replies are the declared result."""
    from tui_gateway.contracts.canonical_sessions import CanonicalSnapshot
    store, owner, home = _owner_home(monkeypatch)
    try:
        first = await _create(owner, home, 'first', title='fresh')
        again = await _create(owner, home, 'again', title='fresh')
        assert 'result' in first and 'result' in again, (first, again)

        assert first['result']['info']['created'] is True
        assert again['result']['info']['created'] is False
        assert again['result']['stored_session_id'] == first['result']['stored_session_id']
        # Both are exactly the declared ``session.create`` result: nothing in ``info`` is undeclared.
        for reply in (first, again):
            CanonicalSnapshot.model_validate(reply['result'])
    finally:
        await owner.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_an_untitled_create_reports_the_row_it_made(monkeypatch):
    """``_create(params, None)``: no title to resolve against, so the create branch always runs and
    every plain CLI start's ``session.create`` reports ``created: true``."""
    from tui_gateway.contracts.canonical_sessions import CanonicalSnapshot
    store, owner, home = _owner_home(monkeypatch)
    try:
        first = await _create(owner, home, 'plain-1')
        second = await _create(owner, home, 'plain-2')
        assert first['result']['info']['created'] is True
        assert second['result']['info']['created'] is True
        assert second['result']['stored_session_id'] != first['result']['stored_session_id']
        CanonicalSnapshot.model_validate(first['result'])
    finally:
        await owner.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_a_retried_request_id_reports_the_row_as_already_made(monkeypatch):
    """An exact retry of a ``request_id`` whose earlier attempt committed: ``stage_local_session``
    returns the restored row, not a staged receipt, so ``_create`` reports ``created: false`` for the
    same session the first attempt made. Fail-safe on purpose — the retrying client is not exempt
    from the resume comparison, so it refuses rather than over-trusts."""
    from tui_gateway.contracts.canonical_sessions import CanonicalSnapshot
    store, owner, home = _owner_home(monkeypatch)
    try:
        first = await _create(owner, home, 'retried')
        retry = await _create(owner, home, 'retried')
        assert first['result']['info']['created'] is True
        assert retry['result']['info']['created'] is False
        assert retry['result']['stored_session_id'] == first['result']['stored_session_id']
        CanonicalSnapshot.model_validate(retry['result'])
    finally:
        await owner.close()
        store.close_all_db_handles()
