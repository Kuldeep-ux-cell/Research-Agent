# Research Agent

Describe what you are working on and the problem you face; Research Agent finds related studies from
OpenAlex (250M+ papers), scores how relevant each one is, and shows exactly how it got there.

- **Describe your problem**: an LLM fixes spelling, maps your own words to research terms (with quotes),
  proposes hypotheses that connect them, and builds combined search queries. Candidates are re-ranked by
  meaning, each with a reason and an evidence quote that is verified against the abstract.
- **Keyword search**: paper cards (citations, Top 1% / Top 10%, open access, PDF links), insights
  (papers per year, top fields, institutions, countries) and an AI brief of the top results.
- **Cost tracking**: every OpenAlex and LLM call is costed and shown in the UI.

See [Research_Agent_Project_Report.pdf](Research_Agent_Project_Report.pdf) for the full design, test results and cost analysis.

## Run

```bash
npm install
cp .env.example .env   # then add your keys
npm start              # http://localhost:3000
```

Environment variables (`.env`, never committed):
- `GROQ_API_KEY` – LLM for problem matching and briefs (model `openai/gpt-oss-120b`).
- `ANTHROPIC_API_KEY` – alternative LLM (Claude), used only if no Groq key is set.
- `OPENALEX_API_KEY` – optional; raises the free $0.10/day OpenAlex allowance.

With no LLM key, problem matching falls back to a keyword heuristic.

## API

| Route | What it does | OpenAlex calls |
|---|---|---|
| `POST /api/match {problem, from}` | Problem text → understand → retrieve → pre-filter → re-rank → scored studies + step trace | 4–6 × $0.001 |
| `GET /api/search?q=&from=&sort=&oa=&cursor=` | 20 normalized papers + next cursor | 1 × $0.001 |
| `GET /api/insights?q=&from=&oa=` | 4 `group_by` aggregations | 4 × $0.0001 |
| `POST /api/brief {q, papers}` | LLM summary of the top 10 abstracts | – |

Responses are cached in memory for 6 hours (repeat searches cost $0).

## Scoring

`score = 65% Meaning (LLM, quote-verified) + 20% Keywords (OpenAlex relevance, query hits, concept coverage) + 15% Impact (citation percentile)`
