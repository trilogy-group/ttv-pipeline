from types import SimpleNamespace

from PIL import Image

import keyframe_generator


def test_gemini_keyframe_generation_uses_google_genai_client(tmp_path, monkeypatch):
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.png"
    Image.new("RGB", (2, 2), color="blue").save(input_path)
    generated_bytes = input_path.read_bytes()

    generate_content_calls = []
    clients = []

    class FakeModels:
        def generate_content(self, **kwargs):
            generate_content_calls.append(kwargs)
            return SimpleNamespace(
                parts=[SimpleNamespace(inline_data=SimpleNamespace(data=generated_bytes))]
            )

    class FakeClient:
        def __init__(self, *, api_key):
            assert api_key == "test-key"
            self.models = FakeModels()
            self.closed = False
            clients.append(self)

        def close(self):
            self.closed = True

    monkeypatch.setattr(keyframe_generator.genai, "Client", FakeClient)

    result = keyframe_generator.generate_keyframe_with_gemini(
        prompt="Turn the square green",
        output_path=str(output_path),
        gemini_api_key="test-key",
        input_image_path=str(input_path),
        model_name="test-image-model",
        max_retries=0,
    )

    assert result == str(output_path.resolve())
    assert output_path.read_bytes() == generated_bytes
    assert generate_content_calls[0]["model"] == "test-image-model"
    assert generate_content_calls[0]["contents"][-1] == "Turn the square green"
    assert generate_content_calls[0]["contents"][0].inline_data.data == input_path.read_bytes()
    assert clients[0].closed
