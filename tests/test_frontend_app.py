import importlib
import sys
import types


class DummyResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception("http error")

    def json(self):
        return self._payload


class _DummyColumn:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def _build_fake_streamlit(mode: str, button_value: bool):
    module = types.ModuleType("streamlit")
    state = {
        "warnings": [],
        "errors": [],
        "metrics": [],
        "bar_chart": 0,
        "dataframes": 0,
    }

    module.sidebar = types.SimpleNamespace(
        text_input=lambda label, value=None: value,
        caption=lambda text: None,
    )

    def cache_data(ttl=60):
        def decorator(func):
            return func

        return decorator

    module.cache_data = cache_data
    module.set_page_config = lambda **kwargs: None
    module.title = lambda text: None
    module.caption = lambda text: None
    module.warning = lambda msg: state["warnings"].append(msg)
    module.radio = lambda label, options, horizontal=False: mode
    module.columns = lambda n: [_DummyColumn() for _ in range(n)]
    module.selectbox = lambda label, options: options[0]
    module.slider = lambda label, min_value=None, max_value=None, value=None: value
    module.button = lambda label, type=None: button_value
    module.metric = lambda label, value: state["metrics"].append((label, value))
    module.error = lambda msg: state["errors"].append(msg)
    module.bar_chart = lambda data: state.__setitem__("bar_chart", state["bar_chart"] + 1)
    module.dataframe = lambda df, use_container_width=True: state.__setitem__("dataframes", state["dataframes"] + 1)

    return module, state


def _build_fake_requests(metadata_payload=None, predict_payload=None, fail_get=False, fail_post=False):
    module = types.ModuleType("requests")

    class RequestException(Exception):
        pass

    module.RequestException = RequestException

    def get(url, timeout=10):
        if fail_get:
            raise RequestException("metadata unavailable")
        return DummyResponse(metadata_payload or {"crops": ["Wheat"], "areas": ["Albania"]})

    def post(url, json=None, timeout=15):
        if fail_post:
            raise RequestException("post failed")
        payload = predict_payload or {"predicted_yield": 123.0, "recommendations": []}
        return DummyResponse(payload)

    module.get = get
    module.post = post
    return module


def _import_frontend_app(monkeypatch, fake_streamlit, fake_requests):
    monkeypatch.setitem(sys.modules, "streamlit", fake_streamlit)
    monkeypatch.setitem(sys.modules, "requests", fake_requests)
    sys.modules.pop("frontend.app", None)
    import frontend.app

    return importlib.reload(frontend.app)


def test_frontend_prediction_flow_and_helpers(monkeypatch) -> None:
    # L'import du module doit permettre le mode prediction et l'appel API de prediction.
    fake_streamlit, state = _build_fake_streamlit(mode="Prediction", button_value=True)
    fake_requests = _build_fake_requests(
        metadata_payload={"crops": ["Wheat"], "areas": ["Albania"]},
        predict_payload={"predicted_yield": 555.0},
    )
    module = _import_frontend_app(monkeypatch, fake_streamlit, fake_requests)

    metadata = module.load_metadata()
    prediction = module.post_json("/predict", {"Item": "Wheat"})

    assert metadata["crops"] == ["Wheat"]
    assert prediction["predicted_yield"] == 555.0
    assert state["metrics"]


def test_frontend_recommendation_fallback_and_error(monkeypatch) -> None:
    # En cas d'API indisponible, l'UI passe en options par defaut et affiche une erreur lisible.
    fake_streamlit, state = _build_fake_streamlit(mode="Recommendation", button_value=True)
    fake_requests = _build_fake_requests(fail_get=True, fail_post=True)
    _import_frontend_app(monkeypatch, fake_streamlit, fake_requests)

    assert state["warnings"]
    assert state["errors"]


def test_frontend_prediction_error_shows_error_message(monkeypatch) -> None:
    # Une erreur API en mode prediction doit etre affichee explicitement.
    fake_streamlit, state = _build_fake_streamlit(mode="Prediction", button_value=True)
    fake_requests = _build_fake_requests(
        metadata_payload={"crops": ["Wheat"], "areas": ["Albania"]},
        fail_post=True,
    )
    _import_frontend_app(monkeypatch, fake_streamlit, fake_requests)

    assert state["errors"]


def test_frontend_recommendation_success_renders_outputs(monkeypatch) -> None:
    # En mode recommandation, l'UI doit tracer un graphique et un tableau classes.
    fake_streamlit, state = _build_fake_streamlit(mode="Recommendation", button_value=True)
    fake_requests = _build_fake_requests(
        metadata_payload={"crops": ["Wheat", "Maize"], "areas": ["Albania"]},
        predict_payload={
            "recommendations": [
                {"Item": "Wheat", "predicted_yield": 10.0},
                {"Item": "Maize", "predicted_yield": 9.0},
            ]
        },
    )
    _import_frontend_app(monkeypatch, fake_streamlit, fake_requests)

    assert state["bar_chart"] >= 1
    assert state["dataframes"] >= 1
