# Common APIs Reference

Pre-tested API examples for the `api-caller` skill.

## JSONPlaceholder (Testing API)

**Base URL:** https://jsonplaceholder.typicode.com

**No authentication required**

### Examples

```bash
# Get all posts
--url "https://jsonplaceholder.typicode.com/posts"

# Get single post
--url "https://jsonplaceholder.typicode.com/posts/1"

# Get comments for a post
--url "https://jsonplaceholder.typicode.com/posts/1/comments"

# Create a post (POST)
--url "https://jsonplaceholder.typicode.com/posts" --method POST --data '{"title": "foo", "body": "bar", "userId": 1}'
```

---

## GitHub API

**Base URL:** https://api.github.com

**Authentication:** Optional for public repos, required for private

### Examples

```bash
# Get repository info
--url "https://api.github.com/repos/owner/repo"

# Get user info
--url "https://api.github.com/users/username"

# With authentication
--url "https://api.github.com/user" --headers '{"Authorization": "Bearer YOUR_TOKEN"}'
```

---

## httpbin (Testing/Debug API)

**Base URL:** https://httpbin.org

**No authentication required**

### Examples

```bash
# Echo GET request
--url "https://httpbin.org/get"

# Echo POST request
--url "https://httpbin.org/post" --method POST --data '{"test": "data"}'

# Get headers
--url "https://httpbin.org/headers"

# Get IP
--url "https://httpbin.org/ip"
```

---

## Tips

1. **Always check API docs** for rate limits and required headers
2. **Use --headers for auth**: `--headers '{"Authorization": "Bearer TOKEN"}'`
3. **For complex workflows**: Use `code-generator` skill instead
