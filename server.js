// Research Agent backend: proxies OpenAlex, normalizes works, caches responses,
// tracks per-request cost, and uses an LLM (Groq or Claude) to understand problems and rank studies.
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const PORT = Number(process.env.PORT || 3000);
const OA = "https://api.openalex.org";
const OA_KEY = process.env.OPENALEX_API_KEY || "";

// ---------- LLM providers (prices in USD per million tokens) ----------
const PROVIDERS = {
  groq: { model: process.env.GROQ_MODEL || "openai/gpt-oss-120b", price: { input: 0.15, output: 0.6 } },
  claude: { model: "claude-opus-5-5", price: { input: 4.0, output: 20.0 } },
};
const PROVIDER = process.env.GROQ_API_KEY ? "groq"
  : (process.env.ANTHROPIC_API_KEY || process.env.ANTHROPIC_AUTH_TOKEN) ? "claude" : null;
const llmLabel = () => PROVIDER ? `${PROVIDER === "groq" ? "Groq" : "Claude"} · ${PROVIDERS[PROVIDER].model}` : null;

const SELECT = [
  "id", "doi", "display_name", "publication_date", "cited_by_count", "fwci",
  "primary_location", "authorships", "open_access", "best_oa_location",
  "primary_topic", "keywords", "counts_by_year", "abstract_inverted_index",
  "is_retracted", "citation_normalized_percentile", "type",
].join(",");

// ---------- tiny TTL cache (OpenAlex updates daily, so hours are fine) ----------
const cache = new Map();
const TTL_MS = 6 * 60 * 60 * 1000;
function cacheGet(k) {
  const hit = cache.get(k);
  if (hit && Date.now() - hit.t < TTL_MS) return hit.v;
  cache.delete(k);
  return null;
}
function cacheSet(k, v) {
  cache.set(k, { t: Date.now(), v });
}

// ---------- OpenAlex ----------
async function openalex(pathAndQuery) {
  const url = new URL(OA + pathAndQuery);
  if (OA_KEY) url.searchParams.set("api_key", OA_KEY);
  const key = url.toString().replace(/api_key=[^&]+/, "");
  const cached = cacheGet(key);
  if (cached) return { ...cached, costUsd: 0, cached: true };

  const res = await fetch(url);
  if (!res.ok) {
    const text = await res.text();
    throw Object.assign(new Error(`OpenAlex ${res.status}: ${text.slice(0, 200)}`), { status: res.status });
  }
  const data = await res.json();
  const out = {
    data,
    costUsd: Number(res.headers.get("x-ratelimit-cost-usd") || 0),
    remainingUsd: res.headers.get("x-ratelimit-remaining-usd"),
  };
  cacheSet(key, out);
  return { ...out, cached: false };
}

function rebuildAbstract(inv) {
  if (!inv) return null;
  const words = [];
  for (const [word, positions] of Object.entries(inv)) for (const p of positions) words[p] = word;
  return words.join(" ");
}

function normalize(w) {
  return {
    id: w.id.split("/").pop(),
    title: w.display_name,
    date: w.publication_date,
    type: w.type,
    journal: w.primary_location?.source?.display_name || null,
    doi: w.doi,
    authors: (w.authorships || []).map((a) => ({
      name: a.author?.display_name,
      orcid: a.author?.orcid,
      inst: a.institutions?.[0]?.display_name || null,
      country: a.institutions?.[0]?.country_code || null,
    })),
    citations: w.cited_by_count,
    fwci: w.fwci,
    top1pct: !!w.citation_normalized_percentile?.is_in_top_1_percent,
    top10pct: !!w.citation_normalized_percentile?.is_in_top_10_percent,
    trend: (w.counts_by_year || []).slice().sort((a, b) => a.year - b.year),
    oa: !!w.open_access?.is_oa,
    oaStatus: w.open_access?.oa_status,
    pdf: w.best_oa_location?.pdf_url || null,
    topic: w.primary_topic?.display_name || null,
    field: w.primary_topic?.field?.display_name || null,
    keywords: (w.keywords || []).map((k) => k.display_name),
    abstract: rebuildAbstract(w.abstract_inverted_index),
    retracted: !!w.is_retracted,
  };
}

// Escape commas/colons that would break OpenAlex filter syntax.
const cleanTerm = (s) => s.replace(/[,:|"]/g, " ").replace(/\s+/g, " ").trim();

function buildFilter({ q, from, oa }) {
  const parts = [`title_and_abstract.search:${cleanTerm(q)}`];
  if (from) parts.push(`from_publication_date:${from}`);
  if (oa === "1") parts.push("is_oa:true");
  return parts.join(",");
}

async function handleSearch(params) {
  const q = params.get("q");
  if (!q) throw Object.assign(new Error("q is required"), { status: 400 });
  const sort = params.get("sort") || "cited_by_count:desc";
  const cursor = params.get("cursor") || "*";
  const qs = new URLSearchParams({
    filter: buildFilter({ q, from: params.get("from"), oa: params.get("oa") }),
    per_page: "20",
    cursor,
    select: SELECT,
  });
  if (sort !== "relevance") qs.set("sort", sort);
  const r = await openalex(`/works?${qs}`);
  return {
    total: r.data.meta.count,
    nextCursor: r.data.meta.next_cursor,
    papers: r.data.results.map(normalize),
    cost: { openalexUsd: r.costUsd, cached: r.cached, remainingUsd: r.remainingUsd },
  };
}

async function handleInsights(params) {
  const q = params.get("q");
  if (!q) throw Object.assign(new Error("q is required"), { status: 400 });
  const filter = buildFilter({ q, from: params.get("from"), oa: params.get("oa") });
  const group = (by) => openalex(`/works?${new URLSearchParams({ filter, group_by: by })}`);
  const [years, countries, fields, insts] = await Promise.all([
    group("publication_year"),
    group("authorships.countries"),
    group("primary_topic.field.id"),
    group("authorships.institutions.lineage"),
  ]);
  const top = (r, n) => r.data.group_by.slice(0, n).map((g) => ({ key: g.key, name: g.key_display_name, count: g.count }));
  return {
    years: r2years(years.data.group_by),
    countries: top(countries, 10),
    fields: top(fields, 8),
    institutions: top(insts, 10),
    cost: {
      openalexUsd: [years, countries, fields, insts].reduce((s, r) => s + r.costUsd, 0),
      calls: 4,
    },
  };
}
function r2years(groups) {
  return groups
    .map((g) => ({ year: Number(g.key), count: g.count }))
    .filter((g) => g.year >= 1990 && g.year <= new Date().getFullYear())
    .sort((a, b) => a.year - b.year);
}

// ---------- LLM call (one interface, two providers) ----------
let anthropic = null;
async function getClaude() {
  if (!anthropic) {
    const { default: Anthropic } = await import("@anthropic-ai/sdk");
    anthropic = new Anthropic();
  }
  return anthropic;
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function groqCall({ system, user, schema, maxTokens, effort }) {
  const body = {
    model: PROVIDERS.groq.model,
    messages: [{ role: "system", content: system }, { role: "user", content: user }],
    max_completion_tokens: maxTokens,
    reasoning_effort: effort,
  };
  if (schema) body.response_format = { type: "json_schema", json_schema: { name: "result", strict: true, schema } };
  for (let attempt = 0; ; attempt++) {
    const res = await fetch("https://api.groq.com/openai/v1/chat/completions", {
      method: "POST",
      headers: { authorization: `Bearer ${process.env.GROQ_API_KEY}`, "content-type": "application/json" },
      body: JSON.stringify(body),
    });
    if (res.status === 429 && attempt < 2) {
      await sleep(Math.min(Number(res.headers.get("retry-after") || 2), 10) * 1000);
      continue;
    }
    const data = await res.json();
    if (!res.ok) throw Object.assign(new Error(`Groq ${res.status}: ${data.error?.message || res.statusText}`), { status: 502 });
    const choice = data.choices[0];
    if (choice.finish_reason === "length") throw new Error("LLM output was truncated (max tokens).");
    return { text: choice.message.content, input: data.usage.prompt_tokens, output: data.usage.completion_tokens };
  }
}

async function claudeCall({ system, user, schema, maxTokens, effort }) {
  const client = await getClaude();
  const response = await client.beta.messages.create({
    model: PROVIDERS.claude.model,
    max_tokens: maxTokens,
    betas: ["server-side-fallback-2026-07-01"],
    fallbacks: "default",
    output_config: { effort, ...(schema && { format: { type: "json_schema", schema } }) },
    system,
    messages: [{ role: "user", content: user }],
  });
  if (response.stop_reason === "refusal") throw Object.assign(new Error("The model declined this request."), { status: 422 });
  if (response.stop_reason === "max_tokens") throw new Error("LLM output was truncated (max tokens).");
  const text = response.content.filter((b) => b.type === "text").map((b) => b.text).join("");
  return { text, input: response.usage.input_tokens, output: response.usage.output_tokens };
}

async function llm({ system, user, schema, maxTokens = 4000, effort = "low" }) {
  if (!PROVIDER) throw Object.assign(new Error("Set GROQ_API_KEY or ANTHROPIC_API_KEY to enable AI features."), { status: 501 });
  const r = PROVIDER === "groq"
    ? await groqCall({ system, user, schema, maxTokens, effort })
    : await claudeCall({ system, user, schema, maxTokens, effort });
  const price = PROVIDERS[PROVIDER].price;
  return {
    text: r.text,
    data: schema ? JSON.parse(r.text) : null,
    tokens: { input: r.input, output: r.output },
    costUsd: (r.input * price.input + r.output * price.output) / 1e6,
  };
}

// ---------- AI brief over the top search results ----------
async function handleBrief(body) {
  const { q, papers } = body;
  if (!q || !Array.isArray(papers) || papers.length === 0) {
    throw Object.assign(new Error("q and papers[] are required"), { status: 400 });
  }
  const corpus = papers.slice(0, 10).map((p, i) =>
    `[${i + 1}] ${p.title} (${p.date}, ${p.journal || "n/a"}, ${p.citations} citations)\n${(p.abstract || "No abstract.").slice(0, 2000)}`
  ).join("\n\n");
  const r = await llm({
    system: "You are a research analyst. Write concise, factual briefs grounded only in the provided abstracts. Cite papers as [n].",
    user: `Research question: "${q}"\n\nPapers:\n${corpus}\n\nWrite a brief with three short sections: "Key findings", "Open problems", "Where the field is heading". Under 250 words. Plain text, no markdown headings beyond those section titles.`,
    maxTokens: 4000,
  });
  return { text: r.text, model: llmLabel(), cost: { inputTokens: r.tokens.input, outputTokens: r.tokens.output, claudeUsd: r.costUsd } };
}

// ---------- Problem → related studies (the Research Agent pipeline) ----------
// 1. Understand: LLM fixes spelling, quotes the user's own phrases, maps each to technical terms,
//    reasons about what could connect them (hypotheses), picks a research domain, and writes queries
//    that combine concepts so results address the whole problem, not one word of it.
// 2. Retrieve:   each query hits OpenAlex in parallel (restricted to the chosen domain); results pooled.
// 3. Pre-filter: cheap lexical score (OpenAlex relevance + multi-query hits + concept coverage).
// 4. Re-rank:    LLM reads the candidates' abstracts, scores relevance 0-100, and must quote evidence
//    from the abstract. The quote is checked against the real text; unverifiable quotes are penalized.
// 5. Combine:    final = 65% meaning (LLM) + 20% keywords + 15% impact (citation percentile).

// OpenAlex top-level domains.
const DOMAINS = { health: 4, life: 1, physical: 3, social: 2 };

const UNDERSTAND_SCHEMA = {
  type: "object",
  additionalProperties: false,
  required: ["corrected_text", "summary", "domain", "key_phrases", "hypotheses", "queries"],
  properties: {
    corrected_text: { type: "string" },
    summary: { type: "string" },
    domain: { type: "string", enum: ["health", "life", "physical", "social", "any"] },
    key_phrases: {
      type: "array",
      items: {
        type: "object",
        additionalProperties: false,
        required: ["quote", "term", "synonyms"],
        properties: {
          quote: { type: "string" },
          term: { type: "string" },
          synonyms: { type: "array", items: { type: "string" } },
        },
      },
    },
    hypotheses: {
      type: "array",
      items: {
        type: "object",
        additionalProperties: false,
        required: ["name", "why"],
        properties: { name: { type: "string" }, why: { type: "string" } },
      },
    },
    queries: {
      type: "array",
      items: {
        type: "object",
        additionalProperties: false,
        required: ["q", "intent"],
        properties: { q: { type: "string" }, intent: { type: "string" } },
      },
    },
  },
};

const UNDERSTAND_SYSTEM = `You are a research librarian who converts a person's informal description of their work or problem into a precise literature-search plan.

Rules:
- The text may contain spelling mistakes, run-together words and lay language. Fix them in corrected_text.
- key_phrases: for each distinct aspect of the problem, copy the user's ORIGINAL words exactly (including their typos) into "quote", put the standard scientific/technical term in "term" (e.g. "hairloss" -> "alopecia"), and give 1-3 synonyms used in academic titles.
- hypotheses: 2-4 mechanisms, causes or solution approaches that could plausibly CONNECT the aspects together (for symptoms: candidate underlying conditions; for engineering: candidate techniques). Each with a one-line "why".
- domain: the research domain the papers should come from (health = medicine/clinical; life = biology/agriculture; physical = engineering/chemistry/physics/CS; social = economics/psychology/education). Use "any" only if truly mixed.
- queries: 4-6 short queries (2-4 words, scientific terminology, no boolean operators, no quotes). Most queries must COMBINE two aspects or an aspect with a hypothesis (e.g. "alopecia weight loss", "hyperthyroidism alopecia"), because single-aspect queries return papers that only cover one part of the problem. "intent" says what the query is meant to find.`;

const RERANK_SCHEMA = {
  type: "object",
  additionalProperties: false,
  required: ["results"],
  properties: {
    results: {
      type: "array",
      items: {
        type: "object",
        additionalProperties: false,
        required: ["i", "score", "reason", "evidence", "covers"],
        properties: {
          i: { type: "integer" },
          score: { type: "integer" },
          reason: { type: "string" },
          evidence: { type: "string" },
          covers: { type: "array", items: { type: "string" } },
        },
      },
    },
  },
};

const RERANK_SYSTEM = `You are an expert research librarian ranking papers for a specific person's problem.
Score each paper 0-100 for how much it helps with THIS problem as a whole:
- 85-100: directly studies the combination of aspects, or a hypothesis that explains them together.
- 60-84: covers one hypothesis or two aspects in a clinically/technically relevant way.
- 30-59: covers a single aspect only, or the right topic in a weakly related setting.
- 0-29: shares words but is off-topic (different organism, field or meaning).
"evidence": copy ONE short exact sentence fragment (8-25 words) from the paper's title or abstract that supports your score. Copy it character-for-character; do not paraphrase.
"covers": which of the listed aspect terms the paper addresses (use the exact term strings given).
"reason": one sentence, plain language, on how it relates to the person's problem.`;

const squash = (s) => String(s || "").toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();

const STOP = new Set(("a an and are as at be but by for from has have i im in into is it its my of on or our that the this to " +
  "we was were what when which while with how why can cant not no do does doing working work trying try problem problems " +
  "issue issues facing face need want using use used get getting very also about between than then there their they them " +
  "these those so such would could should will just like any some more most other over under after before because had").split(" "));
const GENERIC = new Set(("developing develop building build making make small large larger big new good better best low high " +
  "cost costs expensive cheap affordable affordably quickly fast slow straight through pass passes system systems method methods " +
  "approach approaches way ways thing things currently current really still already able unable lot lots many much severe").split(" "));

// No-AI fallback: adjacent content-word pairs make reasonable queries; single words become the concepts.
function heuristicUnderstand(problem) {
  const tokens = problem.toLowerCase().split(/[^a-z0-9-]+/).filter(Boolean);
  const isContent = (w) => w.length > 2 && !STOP.has(w) && !GENERIC.has(w) && !/^\d+$/.test(w);
  const freq = new Map();
  for (const w of tokens) if (isContent(w)) freq.set(w, (freq.get(w) || 0) + 1);
  const wscore = (w) => (freq.get(w) || 0) + Math.min(w.length, 12) / 12;
  const words = [...freq.keys()].sort((a, b) => wscore(b) - wscore(a)).slice(0, 6);
  const bigrams = new Map();
  for (let i = 0; i < tokens.length - 1; i++) {
    const [a, b] = [tokens[i], tokens[i + 1]];
    if (isContent(a) && isContent(b)) bigrams.set(`${a} ${b}`, wscore(a) + wscore(b));
  }
  let queries = [...bigrams.entries()].sort((x, y) => y[1] - x[1]).map(([q]) => q).slice(0, 4);
  if (queries.length < 2 && words.length >= 2) queries.push(`${words[0]} ${words[1]}`);
  if (!queries.length && words[0]) queries = [words[0]];
  return {
    corrected_text: problem, summary: "", domain: "any",
    key_phrases: words.map((w) => ({ quote: w, term: w, synonyms: [] })),
    hypotheses: [],
    queries: queries.map((q) => ({ q, intent: "keyword pair from your text" })),
  };
}

const MATCH_SELECT = SELECT + ",relevance_score";

async function handleMatch(body) {
  const problem = String(body.problem || "").trim();
  if (problem.length < 15) throw Object.assign(new Error("Describe your work and problem in at least a sentence."), { status: 400 });
  const from = body.from || "";
  const ai = !!PROVIDER;
  const trace = [];
  const timed = async (step, fn) => {
    const t = Date.now();
    const out = await fn();
    trace.push({ step, ms: Date.now() - t, ...out.trace });
    return out;
  };

  // 1. Understand
  const u = await timed("Understand the problem", async () => {
    if (!ai) return { plan: heuristicUnderstand(problem), trace: { method: "Keyword heuristic (no LLM key)", costUsd: 0 } };
    const r = await llm({
      system: UNDERSTAND_SYSTEM,
      user: `Person's description:\n"""${problem}"""`,
      schema: UNDERSTAND_SCHEMA,
      maxTokens: 4000,
      effort: "medium",
    });
    return { plan: r.data, trace: { method: `${llmLabel()}: fixed spelling, mapped phrases to terms, built search plan`, costUsd: r.costUsd, tokens: r.tokens } };
  });
  const plan = u.plan;
  // Verify each quoted phrase really came from the user (guards against invented aspects).
  const srcSquash = squash(problem);
  plan.key_phrases = plan.key_phrases.slice(0, 8).map((k) => ({ ...k, verified: srcSquash.includes(squash(k.quote)) }));
  const queries = plan.queries.slice(0, 6).map((q) => ({ ...q, q: cleanTerm(q.q) })).filter((q) => q.q);
  const domainId = DOMAINS[plan.domain];
  const aspectTerms = plan.key_phrases.map((k) => k.term);
  const vocab = plan.key_phrases.flatMap((k) => [k.term, ...k.synonyms]).concat(plan.hypotheses.map((h) => h.name)).map(squash).filter(Boolean);

  // 2. Retrieve
  const pool = new Map();
  await timed("Retrieve candidates from OpenAlex", async () => {
    const responses = await Promise.all(queries.map(({ q }) => {
      const filter = [`title_and_abstract.search:${q}`, from && `from_publication_date:${from}`, domainId && `topics.domain.id:${domainId}`]
        .filter(Boolean).join(",");
      return openalex(`/works?${new URLSearchParams({ filter, per_page: "25", select: MATCH_SELECT })}`);
    }));
    responses.forEach((r, qi) => {
      const max = Math.max(...r.data.results.map((w) => w.relevance_score || 0), 1);
      for (const w of r.data.results) {
        const id = w.id.split("/").pop();
        const norm = (w.relevance_score || 0) / max;
        const prev = pool.get(id);
        if (prev) { prev.hits.add(qi); prev.lexMax = Math.max(prev.lexMax, norm); }
        else pool.set(id, { work: w, hits: new Set([qi]), lexMax: norm });
      }
    });
    return { trace: {
      method: `${queries.length} parallel title+abstract searches${domainId ? `, limited to the ${plan.domain} domain` : ""}`,
      costUsd: responses.reduce((s, r) => s + r.costUsd, 0),
      detail: { queries: queries.map((q) => q.q), perQuery: responses.map((r) => r.data.meta.count), uniqueCandidates: pool.size },
    } };
  });

  // 3. Pre-filter (lexical)
  const candidates = await timed("Pre-filter by keyword score", async () => {
    const scored = [...pool.values()].map((c) => {
      const p = normalize(c.work);
      const text = squash(`${p.title} ${p.abstract || ""}`);
      const overlap = vocab.length ? Math.min(1, vocab.filter((k) => text.includes(k)).length / Math.max(2, plan.key_phrases.length)) : 0;
      const lexical = 0.5 * c.lexMax + 0.25 * (c.hits.size / queries.length) + 0.25 * overlap;
      const impact = c.work.citation_normalized_percentile?.value ?? 0;
      return { paper: p, lexical, overlap, impact, hits: c.hits.size, noAbstract: !p.abstract };
    }).sort((a, b) => (b.lexical - (b.noAbstract ? 0.2 : 0)) - (a.lexical - (a.noAbstract ? 0.2 : 0)));
    const top = scored.slice(0, 30);
    return { list: top, trace: {
      method: "0.5 × OpenAlex relevance + 0.25 × queries matched + 0.25 × concept coverage",
      costUsd: 0,
      detail: { kept: top.length, dropped: scored.length - top.length },
    } };
  });

  // 4. Re-rank (semantic, quote-backed)
  await timed("Re-rank by meaning, with evidence quotes", async () => {
    const list = candidates.list;
    if (!ai || !list.length) {
      list.forEach((c) => { c.semantic = Math.round(c.overlap * 100); });
      return { trace: { method: ai ? "No candidates" : "Concept overlap only (no LLM key)", costUsd: 0 } };
    }
    const docs = list.map((c, i) =>
      `[${i}] ${c.paper.title} (${c.paper.date?.slice(0, 4) || "n.d."})\n${(c.paper.abstract || "No abstract.").slice(0, 650)}`).join("\n\n");
    const context = [
      `Problem (spelling corrected): ${plan.corrected_text}`,
      `Aspect terms: ${aspectTerms.join("; ")}`,
      plan.hypotheses.length ? `Hypotheses linking them: ${plan.hypotheses.map((h) => h.name).join("; ")}` : "",
    ].filter(Boolean).join("\n");
    const r = await llm({
      system: RERANK_SYSTEM,
      user: `${context}\n\nCandidate papers:\n\n${docs}\n\nScore every paper by its index.`,
      schema: RERANK_SCHEMA,
      maxTokens: 12000,
      effort: "low",
    });
    let verified = 0;
    for (const s of r.data.results) {
      const c = list[s.i];
      if (!c) continue;
      const source = squash(`${c.paper.title} ${c.paper.abstract || ""}`);
      const ok = squash(s.evidence).length > 10 && source.includes(squash(s.evidence));
      if (ok) verified++;
      // An evidence quote that isn't really in the abstract means the score may be invented: discount it.
      c.semantic = Math.round(Math.max(0, Math.min(100, s.score)) * (ok ? 1 : 0.75));
      c.reason = s.reason;
      c.evidence = s.evidence;
      c.evidenceVerified = ok;
      c.covers = s.covers.filter((t) => aspectTerms.includes(t));
    }
    list.forEach((c) => { c.semantic ??= 0; });
    return { trace: {
      method: `${llmLabel()} read ${list.length} abstracts; ${verified} evidence quotes verified against the text`,
      costUsd: r.costUsd, tokens: r.tokens,
    } };
  });

  // 5. Combine
  const W = ai ? { semantic: 0.65, lexical: 0.2, impact: 0.15 } : { semantic: 0.5, lexical: 0.35, impact: 0.15 };
  const results = await timed("Combine scores", async () => {
    const out = candidates.list.map((c) => {
      const parts = { semantic: c.semantic, lexical: Math.round(c.lexical * 100), impact: Math.round(c.impact * 100) };
      const score = Math.round(W.semantic * parts.semantic + W.lexical * parts.lexical + W.impact * parts.impact);
      return { ...c.paper, score, parts, reason: c.reason, evidence: c.evidence, evidenceVerified: c.evidenceVerified, covers: c.covers || [] };
    }).sort((a, b) => b.score - a.score).slice(0, 15);
    return { list: out, trace: {
      method: `${W.semantic * 100}% meaning + ${W.lexical * 100}% keywords + ${W.impact * 100}% impact`,
      costUsd: 0,
    } };
  });

  return {
    mode: ai ? "ai" : "heuristic",
    llm: llmLabel(),
    understood: {
      corrected: plan.corrected_text,
      summary: plan.summary,
      domain: plan.domain,
      keyPhrases: plan.key_phrases,
      hypotheses: plan.hypotheses,
      queries,
    },
    weights: W,
    results: results.list,
    trace,
    totalCostUsd: trace.reduce((s, t) => s + (t.costUsd || 0), 0),
  };
}

// ---------- HTTP plumbing ----------
function send(res, status, obj) {
  res.writeHead(status, { "content-type": "application/json" });
  res.end(JSON.stringify(obj));
}
function readBody(req) {
  return new Promise((resolve, reject) => {
    let s = "";
    req.on("data", (c) => { s += c; if (s.length > 1e6) req.destroy(); });
    req.on("end", () => { try { resolve(JSON.parse(s || "{}")); } catch (e) { reject(e); } });
  });
}

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, `http://${req.headers.host}`);
  try {
    if (url.pathname === "/api/search") return send(res, 200, await handleSearch(url.searchParams));
    if (url.pathname === "/api/insights") return send(res, 200, await handleInsights(url.searchParams));
    if (url.pathname === "/api/match" && req.method === "POST") return send(res, 200, await handleMatch(await readBody(req)));
    if (url.pathname === "/api/brief" && req.method === "POST") return send(res, 200, await handleBrief(await readBody(req)));
    if (url.pathname === "/api/config") return send(res, 200, { ai: !!PROVIDER, model: llmLabel() });
    if (url.pathname === "/" || url.pathname === "/index.html") {
      res.writeHead(200, { "content-type": "text/html; charset=utf-8" });
      return fs.createReadStream(path.join(here, "public", "index.html")).pipe(res);
    }
    send(res, 404, { error: "not found" });
  } catch (err) {
    console.error(err);
    send(res, err.status || 500, { error: err.message });
  }
});

server.listen(PORT, () => console.log(`Research Agent on http://localhost:${PORT} · LLM: ${llmLabel() || "none (heuristic mode)"}`));
