"""Graceful degradation: when Ollama is unreachable, state extraction returns
an empty delta instead of crashing — letting the LPCI loop continue with
existing state. This is the contract that lets a long-running experiment survive
a transient ollama hiccup.
"""
from __future__ import annotations

from unittest.mock import patch

import lpci


def test_state_extraction_returns_empty_when_ollama_down():
    state = lpci.SessionState()

    with patch("lpci.urllib.request.urlopen", side_effect=ConnectionError("ollama down")):
        delta = lpci.extract_state_delta(
            state,
            user_message="hello",
            assistant_response="hi",
        )

    assert delta == {}


def test_apply_delta_with_empty_delta_is_noop():
    state = lpci.SessionState()
    state.goal = "original-goal"
    lpci.apply_delta(state, {})
    assert state.goal == "original-goal"
