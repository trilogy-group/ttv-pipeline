from pathlib import Path


def test_nginx_security_and_health_proxy_contract():
    config = (Path(__file__).parents[1] / "config/nginx.conf").read_text()

    assert "return 308 https://$host$request_uri;" in config
    assert "location = /healthz {\n            proxy_pass http://api/healthz;" in config
    assert "proxy_set_header X-Forwarded-Proto $scheme;" in config
