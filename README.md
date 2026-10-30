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

## Fix a site from Claude Desktop

An AI agent looks at a photo, proposes the floor's four corners and the fixed objects, and a screen inside the
chat shows them over the photo. Drag a corner if it is off, press 決定 (Decide), and the site file is written.

Add this to `claude_desktop_config.json` (Settings > Developer > Edit Config), with your own paths, and quit
and reopen Claude Desktop with Cmd+Q:

```json
{
  "mcpServers": {
    "physical-sandbox": {
      "command": "/path/to/physical-sandbox/.venv/bin/psbx",
      "args": ["mcp"]
    }
  }
}
```

Needs the `[mcp]` extra, and `[sam]` to cut out objects. Photos stay on your machine; the display copy of a
photo is kept in `~/.cache/physical-sandbox/` (set `PS_CACHE_DIR` to move it). The site file goes to the
`siteDir` the agent was given, as `site.json`.

## License

Apache-2.0. Copyright Atsushi Kojo.
