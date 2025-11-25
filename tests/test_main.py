from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _import_main_with_stubbed_blenderproc() -> Any:
    # Stub blenderproc before importing main to avoid running inside Blender
    if "blenderproc" not in sys.modules:
        import types

        stub = types.ModuleType("blenderproc")
        python_pkg = types.ModuleType("blenderproc.python")
        utility_pkg = types.ModuleType("blenderproc.python.utility")

        class _DummyUtility:
            @staticmethod
            def get_the_one_node_with_type(*_: Any, **__: Any) -> Any:  # pragma: no cover - stub
                return object()

        def _stdout_redirected() -> Any:  # pragma: no cover - stub
            class _DummyCtx:
                def __enter__(self) -> None:
                    return None

                def __exit__(self, *exc: Any) -> None:
                    return None

            return _DummyCtx()

        utility_pkg.Utility = _DummyUtility  # type: ignore[attr-defined]
        utility_pkg.stdout_redirected = _stdout_redirected  # type: ignore[attr-defined]

        # minimal `types` stand-in
        stub.types = types.SimpleNamespace(  # type: ignore[attr-defined]
            MeshObject=object,
            Light=object,
            Entity=object,
        )

        sys.modules["blenderproc"] = stub
        sys.modules["blenderproc.python"] = python_pkg
        sys.modules["blenderproc.python.utility"] = utility_pkg

        # Also pretend there is a submodule named "Utility" to satisfy
        # `from blenderproc.python.utility.Utility import Utility, stdout_redirected`
        utility_module = types.ModuleType("blenderproc.python.utility.Utility")
        utility_module.Utility = _DummyUtility  # type: ignore[attr-defined]
        utility_module.stdout_redirected = _stdout_redirected  # type: ignore[attr-defined]
        sys.modules["blenderproc.python.utility.Utility"] = utility_module

    module = importlib.import_module("bproc_pubvis.main")
    return importlib.reload(module)


main = _import_main_with_stubbed_blenderproc()


class DummyMesh:
    def __init__(self, has_polygons: bool = True) -> None:
        self.polygons = [1] if has_polygons else []


class DummyObj:
    def __init__(self, has_polygons: bool = True) -> None:
        self._has_polygons = has_polygons

    def get_mesh(self) -> DummyMesh:
        return DummyMesh(self._has_polygons)


def _stub_pipeline(monkeypatch: Any, obj: DummyObj, return_from_depth: DummyObj | None = None) -> dict[str, Any]:
    calls: dict[str, Any] = {}

    def init_renderer(**kwargs: Any) -> None:
        calls["init_renderer"] = kwargs

    def setup_obj(**kwargs: Any) -> DummyObj:
        calls["setup_obj"] = kwargs
        return obj

    def make_camera(**kwargs: Any) -> None:
        calls["make_camera"] = kwargs

    def set_look(**kwargs: Any) -> None:
        calls["set_look"] = kwargs

    def setup_backdrop(**kwargs: Any) -> None:
        calls["setup_backdrop"] = kwargs

    def add_ao(**kwargs: Any) -> None:
        calls["add_ao"] = kwargs

    def make_lights(**kwargs: Any) -> None:
        calls["make_lights"] = kwargs

    def render_color(**kwargs: Any) -> list[str]:
        calls["render_color"] = kwargs
        return ["image"]

    def export_obj(**kwargs: Any) -> None:
        calls["export_obj"] = kwargs

    def render_depth(**kwargs: Any) -> DummyObj | None:
        calls["render_depth"] = kwargs
        return return_from_depth

    def make_animation(**kwargs: Any) -> None:
        calls["make_animation"] = kwargs

    monkeypatch.setattr(main, "init_renderer", init_renderer)
    monkeypatch.setattr(main, "setup_obj", setup_obj)
    monkeypatch.setattr(main, "make_camera", make_camera)
    monkeypatch.setattr(main, "set_look", set_look)
    monkeypatch.setattr(main, "setup_backdrop", setup_backdrop)
    monkeypatch.setattr(main, "add_ambient_occlusion", add_ao)
    monkeypatch.setattr(main, "make_lights", make_lights)
    monkeypatch.setattr(main, "render_color", render_color)
    monkeypatch.setattr(main, "export_obj", export_obj)
    monkeypatch.setattr(main, "render_depth", render_depth)
    monkeypatch.setattr(main, "make_animation", make_animation)

    return calls


def test_run_static_render(tmp_path: Path, monkeypatch: Any) -> None:
    obj = DummyObj(has_polygons=True)
    calls = _stub_pipeline(monkeypatch, obj)

    save_path = tmp_path / "static.png"
    cfg = main.Config(data="suzanne", save=save_path, show=False, depth=False)

    main.run(cfg)

    assert "init_renderer" in calls
    assert "setup_obj" in calls
    assert calls["render_color"]["save"] == save_path
    assert "make_animation" not in calls


def test_run_depth_to_pcd_with_gravity_and_backdrop(tmp_path: Path, monkeypatch: Any) -> None:
    obj = DummyObj(has_polygons=True)
    # depth rendering returns the same object so the rest of the pipeline can continue
    calls = _stub_pipeline(monkeypatch, obj, return_from_depth=obj)

    save_path = tmp_path / "depth.png"
    cfg = main.Config(
        data="suzanne",
        pcd=1024,
        depth="ray_trace",
        gravity=True,
        save=save_path,
        show=False,
        shadow="hard",
        ao=0.7,
    )

    main.run(cfg)

    # depth branch used
    assert "render_depth" in calls
    # ambient occlusion branch used
    assert "add_ao" in calls
    # backdrop setup called because gravity/backdrop enabled
    assert "setup_backdrop" in calls
    # lights configured
    assert "make_lights" in calls


def test_run_animation_turn(monkeypatch: Any, tmp_path: Path) -> None:
    obj = DummyObj(has_polygons=True)
    calls = _stub_pipeline(monkeypatch, obj)

    save_path = tmp_path / "anim.gif"
    cfg = main.Config(
        data="suzanne",
        animate=True,
        save=save_path,
        show=False,
        gravity=False,
        pcd=False,
    )

    main.run(cfg)

    assert "make_animation" in calls
    # when animate is used, render_color is not called
    assert "render_color" not in calls


def test_run_depth_early_return_skips_lighting(monkeypatch: Any, tmp_path: Path) -> None:
    obj = DummyObj(has_polygons=True)
    calls = _stub_pipeline(monkeypatch, obj, return_from_depth=None)

    cfg = main.Config(
        data="suzanne",
        depth="ray_trace",
        save=tmp_path / "depth.png",
        show=False,
    )

    main.run(cfg)

    # render_depth was invoked but returned a falsy object, so we exit early
    assert "render_depth" in calls
    assert "make_lights" not in calls
    assert "setup_backdrop" not in calls


def test_run_disables_gravity_for_point_cloud(monkeypatch: Any, tmp_path: Path) -> None:
    # Non-mesh object (no polygons) triggers gravity disable branch
    obj = DummyObj(has_polygons=False)
    calls = _stub_pipeline(monkeypatch, obj, return_from_depth=obj)

    cfg = main.Config(
        data="suzanne",
        pcd=2048,
        depth=False,
        gravity=True,
        light=0.3,  # exercise float light branch
        save=tmp_path / "pcd.png",
        show=False,
    )

    main.run(cfg)

    assert "setup_backdrop" in calls
    # gravity must be disabled for point clouds
    assert calls["setup_backdrop"]["gravity"] is False


def test_run_tumble_animation_disables_gravity(monkeypatch: Any, tmp_path: Path) -> None:
    obj = DummyObj(has_polygons=True)
    calls = _stub_pipeline(monkeypatch, obj)

    cfg = main.Config(
        data="suzanne",
        animate="tumble",
        gravity=True,
        light="medium",  # exercise string light branch
        save=tmp_path / "tumble.gif",
        show=False,
    )

    main.run(cfg)

    # tumble animation should turn gravity off and use a larger offset
    assert "setup_backdrop" in calls
    assert calls["setup_backdrop"]["gravity"] is False
    assert calls["setup_backdrop"]["offset"][2] < -0.05


def test_run_export_and_verbose_logging(monkeypatch: Any, tmp_path: Path) -> None:
    obj = DummyObj(has_polygons=True)
    calls = _stub_pipeline(monkeypatch, obj)

    save_path = tmp_path / "img.png"
    export_path = tmp_path / "mesh.obj"
    cfg = main.Config(
        data="suzanne",
        save=save_path,
        export=export_path,
        show=False,
        verbose=True,
    )

    main.run(cfg)

    # export path is respected
    assert "export_obj" in calls
    assert calls["export_obj"]["path"] == export_path


def test_main_entrypoint_uses_tyro_cli(monkeypatch: Any, tmp_path: Path) -> None:
    obj = DummyObj(has_polygons=True)
    calls = _stub_pipeline(monkeypatch, obj)

    def fake_cli(_cfg_cls: Any) -> Any:
        return main.Config(data="suzanne", save=tmp_path / "cli.png", show=False)

    monkeypatch.setattr(main.tyro, "cli", fake_cli)

    main.main()

    assert "render_color" in calls
