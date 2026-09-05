# Deploying RTO Sentinel

Everything here is the deployment story for a system that is **still not
production-ready** — see [§ What this does not solve](#6-what-this-does-not-solve).
It closes the gaps that were open at the end of Phase 12 — no authentication,
no rate limiting, no TLS — and then the three that closing those exposed:
unscoped keys, no request audit trail, and a rate limit that only held inside
one worker.

---

## 1. Authentication

`/v1/*` requires an API key. `/health` and `/readiness` do not — a probe that
needs a credential fails during exactly the incident it exists to report, and
neither reveals an order, a score or a secret.

```bash
RTO_API_KEYS=console:sk_live_...,ops:sk_live_...:write
```

Entries are `name:secret:scope` triples. **The name is not a credential** — it
identifies the caller in rate-limit accounting and in the audit trail, so a
leaked key can be revoked without guessing who was using it. An entry without a
name is refused at startup.

### Scopes

| Scope | May do |
|---|---|
| `read` (**the default**) | Every `GET`, and every `POST` that computes without storing: scoring, threshold derivation, economic simulation, agent investigation |
| `write` | The above, plus `POST /v1/decisions` and `POST /v1/decisions/override` — the only two routes that change stored state |

**The scope defaults to `read` when omitted**, which is the whole point. A key
added in a hurry cannot append to the decision log; granting that power is a
deliberate act of typing `:write`. The opposite default would mean every key
ever issued could mutate an append-only audit trail.

A read key on a write route gets **403, not 401**: the caller is who they say
they are, and the credential simply does not carry this power. A 401 would send
them off to re-check a key that is working correctly.

The console is issued a **read** key. It displays risk and lets an operator
investigate; recording an override is an ops action, and ops has its own key.

Callers present the key either way:

```bash
curl -H 'X-API-Key: sk_live_...'            https://api.example.com/v1/orders
curl -H 'Authorization: Bearer sk_live_...' https://api.example.com/v1/orders
```

Generate keys with something that is actually random:

```bash
python -c "import secrets; print('sk_live_' + secrets.token_urlsafe(32))"
```

### It cannot be deployed open by accident

`Settings` **refuses to construct** when `RTO_ENV` names a deployed environment
and `RTO_API_KEYS` is empty. The process does not start. `development` and `test`
are exempt — they are not deployments, and a local console with no key should
work — and `/readiness` reports which state is in force either way, so "open" is
never a surprise:

```json
"authentication": {
  "ready": true,
  "detail": "DISABLED - every /v1 endpoint is open to anyone who can reach this port..."
}
```

### Why API keys, and not OAuth or mTLS

There is no identity provider in this system for OAuth to talk to, and no
certificate infrastructure for mTLS. Building either would have meant shipping a
login screen with nothing behind it. API keys are what this shape of service — a
merchant console plus server-to-server callers — actually uses.

### The browser never holds the key

**A key compiled into a frontend bundle is readable by anyone who opens dev
tools.** Putting one there would look like authentication and protect nothing.

So the console holds no credential. The key is attached by a server-side hop:

```
browser ──▶ reverse proxy (adds X-API-Key) ──▶ API
```

In development the Vite proxy does it (`console/vite.config.ts` reads
`RTO_API_KEY` from the Node process). In production a reverse proxy does the
same. Verified: with the key set, the console works through the proxy, a direct
request without a key gets 401, and `sk_...` appears nowhere in what the browser
receives.

---

## 2. Rate limiting

```bash
RTO_RATE_LIMIT_PER_MINUTE=120   # 0 disables
RTO_RATE_LIMIT_BACKEND=memory   # or 'database'
```

A sliding window per key. Sliding rather than a fixed calendar minute: a fixed
window lets a caller spend its whole allowance at 12:00:59 and again at 12:01:00,
which is twice the intended rate at exactly the moment a burst is most likely.

Limiting is **per key, not per IP**. Per-IP would punish every merchant behind one
corporate NAT for the behaviour of one of them, and would not limit a single key
spread across many addresses.

Exceeding it returns 429 with the delay in the body:

```json
{"error": {"code": "RATE_LIMITED", "detail": {"retry_after_seconds": 41.3}}}
```

### Two backends, and which one you need

| Backend | Counter lives in | Correct for |
|---|---|---|
| `memory` (default) | Process memory | **One** uvicorn worker |
| `database` | PostgreSQL | Any number of workers |

With `memory` and `--workers 4`, each worker keeps its own buckets and the
deployment permits **four times** the configured rate. That was a documented
limitation of this system; `database` removes it.

> **Running more than one worker? Set `RTO_RATE_LIMIT_BACKEND=database`.**
> Nothing warns you at startup, because one worker with `memory` is a correct
> configuration, and a warning that fires on correct setups gets ignored.

PostgreSQL rather than Redis because PostgreSQL is **already deployed here**. A
second datastore to provision, monitor and back up for the sake of one integer
is a poor trade at this scale. Where a Redis already exists, the same interface
would take about thirty lines against it.

The database backend costs one round trip per request — noise beside a scoring
endpoint that takes ~3 s — and stores fixed windows, counting the current window
plus the previous one weighted by how much of it is still in view:

```
estimate = current + previous * (1 - elapsed_fraction)
```

That is the standard sliding-window-counter approximation, and it preserves the
boundary-burst property above. Rows older than two windows are swept
opportunistically on write, because a cron job for three rows is infrastructure
nobody should have to own.

---

## 3. TLS

The application speaks plain HTTP and **should not terminate TLS itself**.
Uvicorn can, but then certificate renewal, cipher policy and HTTP/2 become
application concerns, and the application restarts to pick up a new certificate.

`docker/Caddyfile` terminates TLS, obtains and renews certificates
automatically, injects the API key for the console, and forwards to the app. It
lives in an **overlay file**, not a profile:

```bash
docker compose -f docker-compose.yml -f docker-compose.production.yml up -d
```

Compose interpolates `${VAR:?message}` for every service in a file whatever
profile is selected, so putting the TLS service in the base file made a plain
local `docker compose up` fail asking for a domain — the guard meant to protect
production broke development instead. The overlay keeps both honest: the base
file needs only a database password, and the overlay refuses to start without
the three values a public deployment must have.

Caddy is the default because it gets a certificate with no flags and renews
without a cron entry. Any terminating proxy works; the requirements are TLS, the
`X-API-Key` injection for the console origin, and forwarding `X-Forwarded-Proto`
so the app can tell it is behind one.

---

## 4. The request audit log

One line per `/v1` request, on the `rto_sentinel.access` logger:

```
caller=console scope=read method=GET  path=/v1/orders    query=limit=2&split=test status=200 duration_ms=386.7
caller=console scope=read method=POST path=/v1/decisions query=-                  status=403 duration_ms=68.5
caller=unresolved scope=- method=GET  path=/v1/orders    query=limit=1            status=401 duration_ms=2.3
```

**Why.** A risk system holds every order a merchant has. "Which key read which
order" is the first question asked after a credential leaks, and until now the
answer was that we could not tell — decisions and overrides were logged, but
*reads* were not, and reads are the entire surface a leaked read-only key
exposes.

**The key is never written.** Query strings are, because a filtered order
listing is the thing worth auditing, but neither the `X-API-Key` header nor the
`Authorization` header is touched. A secret in a log is a secret in every
backup, every log shipper, and every screen it is ever displayed on.

Refused requests are recorded too. A run of `caller=unresolved status=401` is
what probing looks like, and it is the pattern most worth having.

Health probes are skipped: unauthenticated, carrying no data, and at one per
second they would bury every line that matters.

It has its own logger so a deployment can route access records separately from
application logs — they have different retention needs and different readers.

---

## 5. A minimal production deployment

```bash
# 1. Secrets. Never commit these.
POSTGRES_PASSWORD=$(python -c "import secrets; print(secrets.token_urlsafe(24))")
RTO_API_KEYS=console:$(python -c "import secrets; print('sk_live_'+secrets.token_urlsafe(32))")
# ...plus an ops key carrying the write scope, if overrides are to be recorded:
#   ops:sk_live_...:write

# 2. Environment
RTO_ENV=production          # refuses to start without RTO_API_KEYS
RTO_CORS_ORIGINS=https://console.example.com   # never '*' - refused with credentials
RTO_DATABASE_URL=postgresql+psycopg://rto:...@db:5432/rto_sentinel
RTO_RATE_LIMIT_PER_MINUTE=120
RTO_RATE_LIMIT_BACKEND=database   # required with more than one worker

# 3. Bring it up
RTO_API_DOMAIN=api.example.com
RTO_CONSOLE_DOMAIN=console.example.com
RTO_CONSOLE_API_KEY=<the console key from RTO_API_KEYS>

docker compose -f docker-compose.yml -f docker-compose.production.yml up -d
docker compose exec api python -m rto_sentinel.cli db upgrade
```

Then confirm the posture rather than assuming it:

```bash
curl -s https://api.example.com/readiness | jq '.components.authentication'
curl -s -o /dev/null -w '%{http_code}\n' https://api.example.com/v1/orders   # expect 401

# And that the console key cannot write:
curl -s -o /dev/null -w '%{http_code}\n' -X POST \
     -H "X-API-Key: $CONSOLE_KEY" -H 'Content-Type: application/json' \
     -d '{"order_id":"ORD-00000001"}' \
     https://api.example.com/v1/decisions                                     # expect 403
```

---

## 6. What this does not solve

Closing six gaps is not the same as being production-ready. What remains:

| Gap | Status |
|---|---|
| **Key rotation** | Manual. Multiple keys can be live at once, so rotation is: add the new key, migrate callers, remove the old one. There is no expiry and no automatic rollover. |
| **Scope granularity** | Two scopes, `read` and `write`. There is no per-route or per-merchant scoping — a read key sees every merchant's orders, because this system serves one merchant. |
| **Audit log retention** | Lines go to a logger. Rotation, retention, shipping and tamper-evidence are the deployment's problem; nothing here guarantees the record survives. |
| **Secret storage** | Keys come from the environment. A real deployment wants a secret manager with rotation and access logging. |
| **The LLM credential** | `ANTHROPIC_API_KEY` is still not configured and no test in this repository executes a real Anthropic call. |
| **Everything in [README § 19](../README.md#19-limitations)** | ~3 s scoring, simulated labels, an economic result whose interval crosses zero. |

**None of the above is a reason to skip what is here.** An unauthenticated API is
a different category of problem from a slow one.
