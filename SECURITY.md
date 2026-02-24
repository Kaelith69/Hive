# Security Policy 🔒

## Supported Versions

Hive is a personal tool / pipeline project. There's currently one version and it's this one.

| Version | Supported |
|---------|-----------|
| 1.x     | ✅ Yes    |
| < 1.0   | ❌ No     |

---

## Reporting a Vulnerability

Found a security issue? Nice catch. Please **do not** open a public GitHub issue for it — that's basically announcing the vulnerability to everyone before it's fixed.

Instead:

1. **Open a private security advisory** on GitHub:
   - Go to the [Security tab](https://github.com/Kaelith69/Hive/security) of this repo
   - Click "Report a vulnerability"
   - Fill in the details

2. **Or email directly** if you prefer. Check the GitHub profile for contact info.

We aim to acknowledge reports within **72 hours** and have a response (fix, workaround, or "this is a feature, you're overthinking it") within **7 days**.

---

## What Counts as a Vulnerability Here

Hive is a local data processing + ML training pipeline. The relevant threat surface:

- **Data exfiltration** — anything that could cause chat data to be sent somewhere without the user's knowledge
- **Malicious JSONL injection** — crafted input files that could cause unsafe code execution
- **Dependency vulnerabilities** — critical CVEs in `torch`, `transformers`, `unsloth`, `peft`, or `trl`
- **Path traversal** in file I/O operations
- **Unsafe deserialization** of model files

If you find any of those: yes, please report it properly.

---

## What Probably Isn't a Vulnerability

- "The model talks like a weird person" — that's the whole point
- "The default model name is hardcoded" — it's configurable, edit the YAML
- "The dataset isn't encrypted at rest" — it's on your own machine, encrypt your drive
- "The training outputs aren't signed" — valid concern for enterprise, not for this project's scope

---

## Privacy Note

This project processes personal chat data. It runs entirely locally (or on your own cloud account). No data is sent to any external service unless you explicitly configure Weights & Biases or TensorBoard. If you believe there's a code path that accidentally exfiltrates data — that is 100% a vulnerability and should be reported.

---

*Professional, but not stiff. We take security seriously even if the project is just vibes-based personality cloning.* 🐝
