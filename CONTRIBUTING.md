# Contributing to Hive 🐝

First off: thanks for even reading this file. Most people skip it. You're already better than average.

Hive is a personal project — a WhatsApp-to-LLM personality pipeline — but contributions are genuinely welcome. If you've got a fix, an improvement, or a feature idea that isn't completely unhinged, let's talk.

---

## 🧭 Branching Model

We keep it simple:

```
main          ← stable, working, the stuff you'd show your parents
feature/*     ← new things (feature/multi-person-support)
fix/*         ← bugs (fix/extract-regex-edge-case)
docs/*        ← documentation only (docs/wiki-update)
```

Always branch off `main`. Never push directly to `main`. That's it. That's the entire branching model.

```bash
git checkout main
git pull origin main
git checkout -b feature/your-cool-thing
```

---

## ✍️ Commit Style

We loosely follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat:     new feature
fix:      bug fix
docs:     documentation only
refactor: code change that isn't a fix or feature
chore:    maintenance (deps, config, tooling)
test:     adding or fixing tests
```

**Examples:**

```bash
git commit -m "feat: add multi-chat merge support in convert.py"
git commit -m "fix: handle empty lines in extract.py regex"
git commit -m "docs: update Installation wiki page"
git commit -m "refactor: simplify format_prompts in train.py"
```

One commit per logical change. Don't squash everything into `git commit -m "stuff"`. That's a cry for help, not a commit message.

---

## 🔁 Pull Request Process

1. Fork the repo
2. Create your branch (`git checkout -b feature/something-cool`)
3. Make your changes
4. Test that nothing explodes (`python extract.py`, `python validate_dataset.py`)
5. Push your branch (`git push origin feature/something-cool`)
6. Open a PR against `main`
7. Describe what you changed and why
8. Wait for review (response time: "when caffeine levels allow")

**PR title format:** same as commit style — `feat: short description`

---

## 🧪 Testing Your Changes

There's no formal test suite yet (it's a personal project, not a bank). But before opening a PR:

- Run `python extract.py` on a sample `.txt` file
- Run `python convert.py` and check the JSONL output
- Run `python validate_dataset.py` to confirm format integrity
- If you touched `train.py`, at minimum verify it parses `training_config.yaml` without crashing

---

## 💡 What Kinds of Contributions Are Welcome

**Yes please:**
- Bug fixes (especially edge cases in parsing)
- Performance improvements to the pipeline
- New export format support
- Better error messages (they can always be funnier)
- Documentation improvements
- Support for more chat export formats (Telegram? Signal?)
- Quality filtering improvements

**Maybe, with discussion:**
- Major architecture changes
- Adding new heavy dependencies
- Changing the default model

**Hard no:**
- Anything that sends user data to a third party
- Anything that breaks the existing pipeline for existing users

---

## 🎨 Code Style

- Python 3.10+ style
- Keep it readable — this isn't code golf
- Comments are welcome when the logic isn't obvious
- Match the existing style of the file you're editing
- No `import *` unless you want to explain yourself

---

## 🤝 Code of Conduct

Be a normal human. Don't be weird. That's literally the entire code of conduct.

---

Questions? Open an issue. Ideas? Open an issue. Found a bug? Open an issue and then also open a PR. You got this. 🐝
