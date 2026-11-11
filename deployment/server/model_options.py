"""Shared model names and service addresses; no model or robot imports."""
from urllib.parse import urlsplit, urlunsplit


MODELS = {'st2nav': {'name': 'ST2Nav', 'source': 'ST2Nav', 'port': 18766, 'checkpoint': 'ckpt.iter0.pth'}}


def model_id(value):
    result = value.lower().replace('-', '')
    if result not in MODELS:
        raise ValueError('This release supports only ST2Nav')
    return result


def model_url(base, selected):
    if selected == 'etpr1':
        return base
    parsed = urlsplit(base)
    host = '[' + parsed.hostname + ']' if ':' in parsed.hostname else parsed.hostname
    return urlunsplit((parsed.scheme, host + ':' + str(MODELS[selected]['port']),
                       parsed.path, parsed.query, parsed.fragment))


def service_health(root, selected):
    import json
    import ssl
    import urllib.request
    from pathlib import Path
    base = json.loads((root / 'config.json').read_text())['model_url']
    context = ssl.create_default_context(cafile=str(Path.home() / '.local/share/etpr1/server.crt'))
    with urllib.request.urlopen(model_url(base, selected) + '/health', context=context, timeout=5) as response:
        result = json.load(response)
    if result.get('ready') is not True or result.get('model_id') != selected:
        raise RuntimeError('Selected model service is not ready: ' + selected)
    return result
