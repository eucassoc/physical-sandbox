# physical-sandbox

Overlay robots on photos and videos of a real site, at real scale. Fix a site from a single picture
(four floor corners, or a few fixed points), then draw robot motions through that camera and compare
two motions with numbers and a replay video. The command is `psbx`.

Everything runs on your machine: photos, videos and cut-outs never leave it.

## Install

```sh
python3.12 -m venv .venv
.venv/bin/pip install -e '.[dev]'          # the core
.venv/bin/pip install -e '.[sam,mcp,dev]'  # plus SAM 2.1 and the MCP server
```

Python 3.10 or newer, macOS or Linux.

## License

Apache-2.0. Copyright Atsushi Kojo.
