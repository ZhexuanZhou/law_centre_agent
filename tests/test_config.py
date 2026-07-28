from pathlib import Path

from dotenv import dotenv_values

from legal_agentic_retrieval.config import ModelConfig


ROOT = Path(__file__).resolve().parents[1]


def test_example_uses_dcguard_qa_deepseek_endpoint_without_a_secret():
    values = dotenv_values(ROOT / ".env.example")

    assert values["LLM_BINDING_HOST"] == "https://api.deepseek.com"
    assert values["LLM_MODEL"] == "deepseek-v4-flash"
    assert values["LLM_BINDING_API_KEY"] == ""
    assert values["OPENAI_LLM_EXTRA_BODY"] == '{"thinking":{"type":"disabled"}}'


def test_model_config_loads_deepseek_settings_from_env_file(tmp_path, monkeypatch):
    example = ROOT / ".env.example"
    values = dotenv_values(example)
    for name in values:
        monkeypatch.delenv(name, raising=False)

    env_file = tmp_path / ".env"
    env_file.write_text(
        example.read_text(encoding="utf-8").replace(
            "LLM_BINDING_API_KEY=\n",
            "LLM_BINDING_API_KEY=test-shared-qa-key\n",
            1,
        ),
        encoding="utf-8",
    )

    config = ModelConfig.from_env(env_file)

    assert config.llm_host == "https://api.deepseek.com"
    assert config.llm_model == "deepseek-v4-flash"
    assert config.llm_api_key == "test-shared-qa-key"
    assert config.llm_extra_body == {"thinking": {"type": "disabled"}}
