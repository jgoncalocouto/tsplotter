from scenario_state import apply_widget_state, collect_widget_state


def test_collect_widget_state_includes_histogram_selection():
    session_state = {
        "hist_cols_main": ["colA", "colB"],
        "hist_binmode": "Fixed width",
        "hist_width_input": 2.5,
        "unrelated": "value",
    }

    collected = collect_widget_state(session_state)

    assert collected == {
        "hist_cols_main": ["colA", "colB"],
        "hist_binmode": "Fixed width",
        "hist_width_input": 2.5,
    }


def test_apply_widget_state_overwrites_existing_values():
    widget_state = {
        "hist_cols_main": ["colA"],
        "sc_x": "colX",
    }
    session_state = {
        "hist_cols_main": ["colB"],
    }

    apply_widget_state(widget_state, session_state)

    assert session_state["hist_cols_main"] == ["colA"]
    assert session_state["sc_x"] == "colX"
