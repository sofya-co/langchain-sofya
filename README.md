# langchain-sofya

LangChain tools for [Sofya](https://sofya.co): live web search, any page or document as clean markdown, AI extraction from a page, and multi-source research with citations. One API key for all four.

## Install

```bash
pip install -U langchain-sofya
```

Get an API key at [sofya.co/dashboard](https://sofya.co/dashboard) and set it:

```bash
export SOFYA_API_KEY="sofya_live_..."
```

Or pass `api_key="sofya_live_..."` to any tool.

## Tools

| Tool | Name the model sees | What it does | Credits |
|---|---|---|---|
| `SofyaSearch` | `sofya_search` | Web search. Results with the text of each page, or snippets only, plus an optional short AI answer. Filter by domain, topic (general or news) and freshness. | 2 (basic), 1 (snippets), +2 for the answer |
| `SofyaFetch` | `sofya_fetch` | Up to 10 URLs as clean markdown. Web pages plus PDF, Word, Excel, PowerPoint, OpenDocument, EPUB and more. | 1 per URL, failed URLs free |
| `SofyaExtract` | `sofya_extract` | Reads one page and answers a prompt about it. | 10 |
| `SofyaResearch` | `sofya_research` | Splits a question into several searches, reads the sources and writes a report with citations. | 50 |

A credit is $0.0025. Pricing: [sofya.co/pricing](https://sofya.co/pricing).

## Use

```python
from langchain_sofya import SofyaSearch

search = SofyaSearch(max_results=5, include_answer=True)
result = search.invoke({"query": "latest stable Python release"})
print(result["answer"])
for r in result["results"]:
    print(r["title"], r["url"])
```

Arguments set on the tool are defaults; the model can override them per call (for example `max_results`, `include_domains` or `freshness`).

```python
from langchain_sofya import SofyaExtract, SofyaFetch, SofyaResearch

SofyaFetch().invoke({"urls": ["https://example.com/report.pdf"]})

SofyaExtract().invoke({
    "url": "https://en.wikipedia.org/wiki/Eiffel_Tower",
    "prompt": "How tall is it and when did it open?",
})

SofyaResearch(max_sources=15).invoke({"query": "How do heat pumps perform in cold climates?"})
```

Every tool also has an async version through `ainvoke`.

### With an agent

```python
from langchain.agents import create_agent
from langchain_sofya import SofyaExtract, SofyaFetch, SofyaSearch

agent = create_agent(
    "anthropic:claude-sonnet-5-5",
    tools=[SofyaSearch(), SofyaFetch(), SofyaExtract()],
)
agent.invoke({"messages": [{"role": "user", "content": "What changed in the latest Node.js LTS?"}]})
```

Errors (invalid key, out of credits, rate limits) come back to the model as a tool message instead of raising, so an agent can tell the user what happened.

## Settings

| Argument | Env var | Default |
|---|---|---|
| `api_key` | `SOFYA_API_KEY` | required |
| `base_url` | `SOFYA_BASE_URL` | `https://sofya.co` |
| `timeout` | - | 120 seconds (research can take up to a minute) |

## Links

- API docs: [sofya.co/docs](https://sofya.co/docs)
- MCP server, for clients that speak MCP instead: `https://mcp.sofya.co/mcp`
- Support: support@sofya.co

## License

MIT
