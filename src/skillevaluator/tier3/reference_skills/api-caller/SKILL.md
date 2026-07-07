---
name: api-caller
description: Call any REST API dynamically. Make GET, POST, PUT, DELETE requests to any endpoint with custom headers and JSON body.
metadata:
  author: Skill Evaluator Maintainers <maintainers@example.com>
---

# API Caller Skill

Generic skill to call any REST API dynamically. Can work with OpenAPI specs or direct endpoint calls.

## Capabilities

- Call any REST API endpoint (GET, POST, PUT, DELETE)
- Parse OpenAPI/Swagger specs to discover endpoints
- Handle common authentication (API keys, Bearer tokens)
- JSON request/response handling

## Instructions

1. Read this SKILL.md to choose the direct-call or OpenAPI workflow.
2. Use `scripts/parse_openapi.py` first when the user gives an OpenAPI or Swagger spec.
3. Use `scripts/call_api.py` for the actual HTTP request.
4. Return the request result and any recoverable error details to the user.

## Scripts

### scripts/parse_openapi.py

**Use this first** when given an API spec URL. Parses an OpenAPI/Swagger spec and returns available operations with parameters. Run commands from the skill base directory; script paths are relative to this SKILL.md file.

**Usage:**
```
python scripts/parse_openapi.py <spec_url> [filter_path]
```

**Arguments:**
- `<spec_url>` - URL to the OpenAPI spec (required)
- `[filter_path]` - Optional: filter operations by path substring

**Example:**
```bash
python scripts/parse_openapi.py "https://api.example.com/openapi.json"
```

Returns: API name, base URL, and list of operations with their parameters, request body schema, etc.

### scripts/call_api.py

Call a REST API endpoint directly or using an OpenAPI spec. Run commands from the skill base directory; script paths are relative to this SKILL.md file.

**Usage:**
```
python scripts/call_api.py --url <endpoint_url> [options]
```

**Arguments:**
- `--url <endpoint_url>` - Direct API endpoint URL (required if no --spec)
- `--method <GET|POST|PUT|DELETE>` - HTTP method (default: GET)
- `--data <json_string>` - Request body as JSON string
- `--headers <json_string>` - Custom headers as JSON string
- `--spec <openapi_url>` - OpenAPI spec URL (optional, for discovery)
- `--operation <operation_id>` - Operation ID from OpenAPI spec
- `--params <json_string>` - Path/query parameters as JSON

## Examples

### Direct API Call (Simple)

```bash
python scripts/call_api.py --url "https://api.example.com/data" --method GET
```

### With Query Parameters

```bash
# Call API with query parameters appended to URL
python scripts/call_api.py --url "https://en.wikipedia.org/w/api.php" --params '{"action": "query", "titles": "NVIDIA", "format": "json"}'
```

### With Headers (API Key)

```bash
# Call API with authentication
python scripts/call_api.py --url "https://api.example.com/data" --headers '{"Authorization": "Bearer TOKEN"}'
```

### POST with JSON Body

```bash
# Create a resource
python scripts/call_api.py --url "https://api.example.com/items" --method POST --data '{"name": "test", "value": 123}'
```

### Using OpenAPI Spec (Advanced)

```bash
# Discover and call an endpoint from OpenAPI spec
python scripts/call_api.py --spec "https://api.example.com/openapi.json" --operation "getItems" --params '{"param": "value"}'
```

## Notes

- For APIs requiring authentication, pass API keys via `--headers`
- The skill automatically parses JSON responses
- Timeout is 30 seconds by default
- For complex API workflows, consider using `code-generator` skill instead
- **Limiting results**: Try common params like `?_limit=N`, `?limit=N`, `?per_page=N`. If unsupported, fetch all and filter with `code-generator`

## Workflow: Using an API Spec

When given an OpenAPI spec URL, follow this workflow:

1. **Parse the spec** first with `python scripts/parse_openapi.py`:
   ```bash
   python scripts/parse_openapi.py "<spec_url>"
   ```

2. **Review operations** - Look at the returned operations, their paths, methods, and parameters

3. **Call the API** with `python scripts/call_api.py` using the discovered endpoint and parameters

## When to Use

- Quick API calls without writing code
- Exploring new APIs
- Simple GET requests for data retrieval
- When you know the exact endpoint URL
- **When given an API spec to explore**

## When NOT to Use

- Complex multi-step API workflows (use `code-generator`)
- APIs requiring OAuth flows
- When you need to process response data extensively
