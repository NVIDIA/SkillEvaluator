# Third-Party Notices

This file lists direct third-party Python dependencies and vendored test
fixtures in this distribution. `pyproject.toml` defines the dependency groups,
and `uv.lock` records the exact resolved dependency set used for this release.

| Dependency group | Packages and licenses |
| --- | --- |
| Base | Click (BSD-3-Clause), Jinja2 (BSD-3-Clause), Pydantic (MIT), PyYAML (MIT), Rich (MIT) |
| LLM | Anthropic (MIT), Boto3 (Apache-2.0), LiteLLM (MIT), OpenAI (Apache-2.0) |
| Tier 3 | Harbor (Apache-2.0) |
| Security | Bandit (Apache-2.0), pip-audit (Apache-2.0) |

## OpenClaw agent-skills test fixture

`tests/fixtures/openclaw-autoreview/` contains pinned files from
`openclaw/agent-skills`, commit `2a409d348a4bcf6f15e41e9a20efd0b298a32528`,
path `skills/autoreview`. The source repository is licensed under the MIT
License:

Copyright (c) 2026 openclaw

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
