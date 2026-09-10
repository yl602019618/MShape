"""Behavioural checks for the MiShape generator's physical public contract."""
import json

import numpy as np
import pytest

from mishape.generation import BODY_STYLES, generate, generation_schema


@pytest.fixture(scope="module")
def baseline():
    return generate({})


def _vertices(model):
    return np.asarray(model["vertices"]).reshape(-1, 3)


def _faces(model):
    return np.asarray(model["faces"]).reshape(-1, 3)


@pytest.mark.parametrize("style", list(BODY_STYLES))
def test_all_body_styles_realise_their_declared_dimensions_and_pass_surface_screen(style):
    model = generate({"body_style": style})
    p = model["metadata"]["recipe"]["parameters"]
    measure = model["metadata"]["measurements"]
    extent = np.ptp(_vertices(model), axis=0)
    assert extent[0] == pytest.approx(p["length"], abs=1e-6)
    assert extent[2] == pytest.approx(p["height"], abs=3e-5)
    assert measure["wheelbase_m"] == pytest.approx(p["wheelbase"], abs=1e-8)
    assert model["metadata"]["quality"]["screen_pass"], model["metadata"]["quality"]["errors"]
    assert model["metadata"]["quality"]["topology"]["closed_manifold_driver"]
    assert model["metadata"]["production_ready"] is False


def test_length_and_wheel_radius_change_independently_without_stretching_tyres(baseline):
    changed = generate({"length": 5.04, "wheelbase": 3.02, "wheel_radius": .38, "cabin_length": 1.50})
    assert np.ptp(_vertices(changed)[:, 0]) == pytest.approx(5.04, abs=1e-6)
    vertices, faces = _vertices(changed), _faces(changed)
    labels = np.asarray(changed["face_labels"])
    centres = {}
    for part in changed["parts"]:
        if not part["key"].startswith("wheel_"):
            continue
        wheel = vertices[np.unique(faces[labels == part["id"]])]
        extent = np.ptp(wheel, axis=0)
        assert extent[0] == pytest.approx(.76, abs=1e-8)
        assert extent[2] == pytest.approx(.76, abs=1e-8)
        assert wheel[:, 2].min() == pytest.approx(0., abs=1e-9)
        centres[part["key"]] = (wheel.max(axis=0) + wheel.min(axis=0)) / 2
    assert centres["wheel_bl"][0] - centres["wheel_fl"][0] == pytest.approx(3.02, abs=1e-8)
    assert not np.array_equal(_vertices(baseline), vertices)


def test_fine_hood_and_greenhouse_controls_change_the_target_surface(baseline):
    hood_model = generate({"hood_crown": .035})
    glass_model = generate({"greenhouse_taper": 8.})
    vertices, faces = _vertices(baseline), _faces(baseline)
    labels = np.asarray(baseline["face_labels"])
    hood_id = next(p["id"] for p in baseline["parts"] if p["key"] == "hood")
    hood_vertices = np.unique(faces[labels == hood_id])
    assert np.max(np.abs(_vertices(hood_model)[hood_vertices, 2] - vertices[hood_vertices, 2])) > .01
    roof_id = next(p["id"] for p in baseline["parts"] if p["key"] == "roof")
    roof_vertices = np.unique(faces[labels == roof_id])
    assert np.max(np.abs(_vertices(glass_model)[roof_vertices, 1] - vertices[roof_vertices, 1])) > .02
    np.testing.assert_array_equal(_faces(hood_model), faces)
    np.testing.assert_array_equal(hood_model["face_labels"], baseline["face_labels"])


def test_component_choice_changes_exported_mesh_and_preserves_materials(baseline):
    no_spoiler = generate({"spoiler_style": "none", "mirror_style": "compact"})
    spoiler = next(p for p in no_spoiler["parts"] if p["key"] == "spoiler")
    assert spoiler["face_count"] == 0 and not spoiler["present"]
    assert spoiler["id"] not in no_spoiler["face_labels"]
    assert len(no_spoiler["faces"]) < len(baseline["faces"])
    assert no_spoiler["metadata"]["component_count"] == 29
    assert len(no_spoiler["parts"]) == 30
    labels = np.asarray(baseline["face_labels"])
    mats = np.asarray(baseline["face_materials"])
    wheel = next(p for p in baseline["parts"] if p["key"] == "wheel_fl")
    assert len(np.unique(mats[labels == wheel["id"]])) >= 3
    for model in [baseline, no_spoiler]:
        nfaces = len(model["faces"]) // 3
        assert len(model["face_labels"]) == len(model["face_materials"]) == len(model["source_face_ids"]) == nfaces
        assert all(p["separable"] for p in model["parts"])


def test_saved_recipe_replays_exact_geometry_and_has_no_hidden_seed_noise(baseline):
    recipe = json.loads(json.dumps(baseline["metadata"]["recipe"]))
    replay = generate(recipe)
    np.testing.assert_array_equal(replay["vertices"], baseline["vertices"])
    np.testing.assert_array_equal(replay["faces"], baseline["faces"])
    assert replay["metadata"]["recipe_hash"] == baseline["metadata"]["recipe_hash"]
    assert replay["metadata"]["geometry_hash"] == baseline["metadata"]["geometry_hash"]


def test_paint_changes_do_not_mutate_shared_template_or_semantic_materials(baseline):
    painted = generate({"paint_color": "#bd75f7"})
    assert any(m["base_color"] == "#bd75f7" for m in painted["materials"])
    assert all(m["base_color"] != "#bd75f7" for m in baseline["materials"])
    np.testing.assert_array_equal(painted["vertices"], baseline["vertices"])
    wheel = next(p for p in painted["parts"] if p["key"] == "wheel_fl")
    assert wheel["material"]["base_color"] != "#bd75f7"


@pytest.mark.parametrize("payload,match", [
    ({"length": 3.9, "wheelbase": 3.2}, "overhang"),
    ({"track": 1.88, "width": 1.70}, "轮距"),
    ({"hood_crown": float("nan")}, "finite"),
    ({"unknown_knob": 4}, "Unknown generation parameters"),
    ({"mirror_style": "invisible"}, "mirror_style"),
    ({"seed": -1}, "seed"),
])
def test_impossible_or_unrecognised_parameters_fail_explicitly(payload, match):
    with pytest.raises(ValueError, match=match):
        generate(payload)


def test_schema_is_self_contained_and_controls_cover_presets():
    schema = generation_schema()
    keys = {f["id"] for f in schema["fields"]}
    assert {"length", "width", "height", "wheelbase", "track", "wheel_radius", "hood_crown", "backlight_angle", "diffuser_style"} <= keys
    assert {f["id"] for g in schema["groups"] for f in g["fields"]} == keys
    for values in schema["presets"].values():
        assert set(values) == keys
    json.dumps(schema, allow_nan=False)
