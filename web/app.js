// Octopus web app: chat, coding mode and the Train section.

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const h = (tag, attrs = {}, ...kids) => {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") el.className = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (v === true) el.setAttribute(k, "");
    else if (v !== false && v != null) el.setAttribute(k, v);
  }
  for (const kid of kids.flat()) if (kid != null) el.append(kid.nodeType ? kid : document.createTextNode(kid));
  return el;
};
const svg = (paths, cls = "") => {
  const el = document.createElement("span");
  el.innerHTML = `<svg viewBox="0 0 24 24" class="${cls}">${paths}</svg>`;
  return el.firstChild;
};
const ICON = {
  copy: '<rect x="8.5" y="8.5" width="11" height="11" rx="2.5"/><path d="M15.5 8.5V6a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v7.5a2 2 0 0 0 2 2h2.5"/>',
  check: '<path d="m5 12.5 4.5 4.5L19 7.5"/>',
  play: '<path d="M8 5.5v13l10.5-6.5Z"/>',
  retry: '<path d="M4 12a8 8 0 1 0 2.4-5.7M4 4.5v4h4"/>',
  trash: '<path d="M5 7h14M10 7V5h4v2M7 7l1 12h8l1-12"/>',
  chev: '<path d="m7 10 5 5 5-5"/>',
  x: '<path d="M7 7l10 10M17 7 7 17"/>',
  file: '<path d="M7 3.5h6.5L18 8v12.5H7Z"/><path d="M13 3.5V8h5"/>',
  wrench: '<path d="M14.5 6.5a4 4 0 0 0 5 5L12 19a2.1 2.1 0 0 1-3-3l7.5-7.5"/>',
  upload: '<path d="M12 16V4M7 9l5-5 5 5M5 20h14"/>',
  spark: '<path d="M12 3v4M12 17v4M3 12h4M17 12h4M6 6l2.5 2.5M15.5 15.5 18 18M6 18l2.5-2.5M15.5 8.5 18 6"/>',
  brain: '<path d="M12 3a4 4 0 0 0-4 4v1a4 4 0 0 0-3 6.5A4 4 0 0 0 9 21h.5M12 3a4 4 0 0 1 4 4v1a4 4 0 0 1 3 6.5A4 4 0 0 1 15 21h-.5M12 3v18"/>',
};

// ---------- tiny storage helpers (never throw) ----------
const store = {
  get(k, d) { try { const v = localStorage.getItem(k); return v == null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch {} },
};

// ---------- API ----------
async function api(path, opts = {}) {
  const res = await fetch(path, {
    method: opts.method || (opts.body ? "POST" : "GET"),
    headers: opts.body ? { "Content-Type": "application/json" } : {},
    body: opts.body ? JSON.stringify(opts.body) : undefined,
    credentials: "same-origin",
  });
  if (res.status === 401 && path !== "/api/login") { showLogin(); throw new Error("Please sign in"); }
  const data = res.headers.get("content-type")?.includes("json") ? await res.json() : await res.text();
  if (!res.ok) throw new Error(data?.detail || data?.error || failText(res.status));
  return data;
}

// When the browser loses its connection to Octopus mid-job, find out why and, if Octopus is still
// running, wait for the job and pick up its result. Returns {asset} or {error}.
// 502, 503 and 504 come from a proxy in front of Octopus (a codespace or a cloud server) when Octopus itself isn't answering.
function failText(status) {
  return [502, 503, 504].includes(status) ? "Octopus's server isn't answering right now. It restarts by itself, so wait a few seconds and press Try again."
    : `Request failed (${status})`;
}
const NET_ERR = /NetworkError|Failed to fetch|network error|Load failed|input stream|body stream/i;
async function recoverStudio(label) {
  for (let i = 0; i < 1200; i++) {
    let st = null;
    try { st = await (await fetch("/healthz", { cache: "no-store" })).json(); } catch {}
    if (!st) return { error: "Octopus stopped running on your computer, so the job ended. Its window (start.bat) was closed or Octopus crashed. "
      + "Double-click start.bat to start it again. If it crashed, the window and the file workspace/octopus.log show why." };
    if (!st.busy) {
      if (st.studio?.asset) return { asset: st.studio.asset };
      return { error: st.studio?.error || "The connection to Octopus dropped. Press Try again." };
    }
    if (label) label.textContent = "Reconnected. Still working on it…";
    await new Promise((r) => setTimeout(r, 3000));
  }
  return { error: "This is taking unusually long. Check the Octopus window for errors." };
}

async function* sse(res) {
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let i;
    while ((i = buf.indexOf("\n\n")) >= 0) {
      const chunk = buf.slice(0, i); buf = buf.slice(i + 2);
      for (const line of chunk.split("\n")) if (line.startsWith("data: ")) yield JSON.parse(line.slice(6));
    }
  }
}

// ---------- state ----------
const S = {
  cfg: null,
  chats: [],
  currentId: null,
  mode: store.get("octopus.mode", "chat"),
  choice: store.get("octopus.choice", { base: null, brain: null }),
  attachments: [],
  streaming: null, // { id, controller }
  view: "chat",
};
// Chats live on the Octopus server under your account, so each person sees only their own.
let chatTimer = 0;
function saveChats() {
  clearTimeout(chatTimer);
  chatTimer = setTimeout(pushChats, 400);
}
async function pushChats() {
  chatTimer = 0;
  if (!S.cfg?.user) return;
  const body = JSON.stringify(S.chats.filter((c) => c.messages.length).slice(0, 300));
  try { await fetch("/api/me/chats", { method: "PUT", headers: { "Content-Type": "application/json" }, body, credentials: "same-origin", keepalive: body.length < 60000 }); }
  catch {}
}
addEventListener("pagehide", () => { if (chatTimer) { clearTimeout(chatTimer); pushChats(); } });
async function loadChats() {
  let chats = await api("/api/me/chats").catch(() => []);
  // First sign-in after updating: the owner gets the chats the old version kept in this browser.
  const old = store.get("octopus.chats", null);
  if (old?.length && S.cfg.user?.role === "owner") {
    const have = new Set(chats.map((c) => c.id));
    chats = [...chats, ...old.filter((c) => !have.has(c.id))];
    store.set("octopus.chats", []);
    S.chats = chats; pushChats();
  }
  S.chats = chats;
}
const current = () => S.chats.find((c) => c.id === S.currentId);
const uid = () => Math.random().toString(36).slice(2) + Date.now().toString(36);

// ---------- theme ----------
function applyTheme(t) {
  if (t === "light" || t === "dark") document.documentElement.dataset.theme = t;
  else delete document.documentElement.dataset.theme;
}
applyTheme(store.get("octopus.theme", "system"));
$("#theme-toggle").addEventListener("click", () => {
  const order = ["system", "light", "dark"];
  const next = order[(order.indexOf(store.get("octopus.theme", "system")) + 1) % 3];
  store.set("octopus.theme", next); applyTheme(next); themeLabel();
});
function themeLabel() { const t = store.get("octopus.theme", "system"); $("#theme-label").textContent = t[0].toUpperCase() + t.slice(1); }
themeLabel();

let toastTimer;
function toast(msg) {
  const t = $("#toast"); t.textContent = msg; t.hidden = false;
  clearTimeout(toastTimer); toastTimer = setTimeout(() => (t.hidden = true), 2200);
}

// ---------- boot & login ----------
async function boot() {
  try {
    S.cfg = await api("/api/config");
  } catch (e) {
    document.body.append(h("p", { class: "error", style: "position:relative;z-index:2;padding:20px" }, "Couldn't reach Octopus: " + e.message));
    return;
  }
  if (!S.cfg.authed) return showLogin(S.cfg.needsSetup ? "setup" : "signin");
  showApp();
}

let authMode = "signin";
function showLogin(mode = "signin") {
  authMode = mode;
  const setup = mode === "setup", signup = mode === "signup" || setup;
  $("#app").hidden = true; $("#login").hidden = false;
  $("#auth-title").textContent = setup ? "Welcome to Octopus" : signup ? "Create your account" : "Welcome back";
  $("#auth-sub").textContent = setup ? "Create the owner account for this Octopus. Everyone who uses it gets their own account, chats and creations."
    : signup ? "Your own chats, 3D models and images, separate from everyone else's." : "Sign in to your Octopus account.";
  $("#auth-name").hidden = !signup;
  $("#auth-code").hidden = !(signup && S.cfg.codeRequired);
  $("#login-password").autocomplete = signup ? "new-password" : "current-password";
  $("#login-password").placeholder = signup ? "Password (6 or more characters)" : "Password";
  $("#auth-submit").textContent = signup ? "Create account" : "Sign in";
  $("#login-error").textContent = "";
  const sw = $("#auth-switch"); sw.replaceChildren();
  if (!setup) {
    if (signup) sw.append("Already have an account? ", h("button", { type: "button", class: "linkish", onclick: () => showLogin("signin") }, "Sign in"));
    else if (S.cfg.signupsOpen) sw.append("New here? ", h("button", { type: "button", class: "linkish", onclick: () => showLogin("signup") }, "Create an account"));
  }
  (signup ? $("#auth-name") : $("#auth-username")).focus();
}
$("#login-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  $("#login-error").textContent = "";
  const signup = authMode !== "signin";
  const body = { username: $("#auth-username").value, password: $("#login-password").value };
  if (signup) Object.assign(body, { name: $("#auth-name").value, code: $("#auth-code").value });
  $("#auth-submit").disabled = true;
  try {
    await api(signup ? "/api/signup" : "/api/login", { body });
    S.cfg = await api("/api/config");
    $("#login-password").value = ""; $("#auth-code").value = "";
    showApp();
  } catch (err) { $("#login-error").textContent = err.message; }
  finally { $("#auth-submit").disabled = false; }
});
$("#logout").addEventListener("click", async () => { clearTimeout(chatTimer); await pushChats(); await api("/api/logout", { body: {} }); location.reload(); });

// ---------- account menu ----------
const accountBtn = $("#account-btn"), accountMenu = $("#account-menu");
function renderAccount() {
  const u = S.cfg.user || {};
  $("#account-name").textContent = u.name || "Account";
  $("#account-role").textContent = u.role === "owner" ? "Owner" : "@" + (u.username || "");
  $("#account-initials").textContent = (u.name || "?").split(/\s+/).map((w) => w[0]).join("").slice(0, 2).toUpperCase();
}
accountBtn.addEventListener("click", (e) => { e.stopPropagation(); accountMenu.hidden = !accountMenu.hidden; accountBtn.setAttribute("aria-expanded", String(!accountMenu.hidden)); });
document.addEventListener("click", (e) => { if (!accountMenu.hidden && !accountMenu.contains(e.target)) { accountMenu.hidden = true; accountBtn.setAttribute("aria-expanded", "false"); } });
accountMenu.addEventListener("click", (e) => { if (e.target.closest("#settings-btn, #logout")) accountMenu.hidden = true; });

async function showApp() {
  $("#login").hidden = true; $("#app").hidden = false;
  renderAccount();
  await loadChats();
  if (!S.choice.base) S.choice.base = S.cfg.defaultBase;
  if (S.choice.brain && !S.cfg.brains.some((b) => b.id === S.choice.brain)) S.choice.brain = null;
  renderModelButton();
  ollamaBanner();
  setMode(S.mode, false);
  loadSettings().then(() => { if (S.mode === "model" || S.mode === "image") setMode(S.mode, false); });
  renderHistory();
  newChat();
  if (S.cfg.training) { $("#train-badge").hidden = false; }
  if (location.hash === "#train") openTrain();
}

// ---------- sidebar ----------
const app = $("#app");
const isMobile = () => matchMedia("(max-width: 860px)").matches;
$("#collapse").addEventListener("click", () => (isMobile() ? app.classList.remove("mobile-open") : app.classList.add("collapsed")));
$("#open-sidebar").addEventListener("click", () => (isMobile() ? app.classList.add("mobile-open") : app.classList.remove("collapsed")));
$("#scrim").addEventListener("click", () => app.classList.remove("mobile-open"));
$("#new-chat").addEventListener("click", () => { newChat(); closeMobile(); });
$("#nav-train").addEventListener("click", () => { openTrain(); closeMobile(); });
$$(".type-item").forEach((b) => b.addEventListener("click", () => {
  if (S.view !== "chat") showChatView();
  setMode(b.dataset.type); newChat(); closeMobile();
}));
$("#search").addEventListener("input", renderHistory);
const closeMobile = () => app.classList.remove("mobile-open");

function renderHistory() {
  const q = $("#search").value.trim().toLowerCase();
  const nav = $("#history"); nav.replaceChildren();
  const list = S.chats.filter((c) => c.messages.length && (!q || c.title.toLowerCase().includes(q) ||
    c.messages.some((m) => m.content.toLowerCase().includes(q))));
  if (!list.length) { nav.append(h("div", { class: "history-empty" }, h("img", { src: "/octo/sleepy.svg", alt: "", "aria-hidden": "true" }), q ? "No chats match." : "Your chats will show up here.")); return; }
  const day = 864e5, now = new Date(), start = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
  const groups = [["Today", start], ["Yesterday", start - day], ["Previous 7 days", start - 7 * day], ["Previous 30 days", start - 30 * day], ["Older", -Infinity]];
  let gi = -1;
  for (const c of list.sort((a, b) => b.updated - a.updated)) {
    const g = groups.findIndex(([, t]) => c.updated >= t);
    if (g !== gi) { gi = g; nav.append(h("h6", {}, groups[g][0])); }
    nav.append(h("div", { class: "chat-item" + (c.id === S.currentId && S.view === "chat" ? " active" : ""), role: "button", tabindex: "0",
      onclick: () => { openChat(c.id); closeMobile(); }, onkeydown: (e) => e.key === "Enter" && openChat(c.id) },
      h("span", { class: "mode-dot m-" + (c.mode || "chat"), title: { chat: "Chat", code: "Code", model: "Modelling", image: "Image" }[c.mode || "chat"] }),
      h("span", { class: "title" }, c.title),
      h("button", { class: "del", title: "Delete chat", "aria-label": "Delete chat", onclick: (e) => { e.stopPropagation(); deleteChat(c.id); } }, svg(ICON.trash))));
  }
}

function deleteChat(id) {
  S.chats = S.chats.filter((c) => c.id !== id); saveChats();
  if (id === S.currentId) newChat(); else renderHistory();
}

// ---------- model picker ----------
const modelBtn = $("#model-btn"), modelMenu = $("#model-menu");
function choiceLabel() {
  if (S.choice.brain) return S.choice.brain;
  return S.cfg.bases.find((b) => b.id === S.choice.base)?.label || "Octopus";
}
function renderModelButton() { $("#model-label").textContent = choiceLabel(); }
modelBtn.addEventListener("click", (e) => { e.stopPropagation(); modelMenu.hidden ? openModelMenu() : closeModelMenu(); });
document.addEventListener("click", (e) => { if (!modelMenu.hidden && !modelMenu.contains(e.target)) closeModelMenu(); });
function closeModelMenu() { modelMenu.hidden = true; modelBtn.setAttribute("aria-expanded", "false"); }
async function openModelMenu() {
  try { S.cfg.brains = await api("/api/brains"); } catch {}
  modelMenu.replaceChildren();
  const item = (label, blurb, selected, onPick, tag) => h("button", { class: "menu-item", role: "option", "aria-selected": String(selected),
    onclick: () => { onPick(); renderModelButton(); closeModelMenu(); } }, h("b", {}, label, tag ? h("span", { class: "size-tag" }, tag) : null), svg(ICON.check, "check"), h("small", {}, blurb));
  if (S.cfg.brains.length) {
    modelMenu.append(h("div", { class: "menu-note" }, "Brains you trained"));
    for (const b of S.cfg.brains) {
      const base = S.cfg.bases.find((x) => x.id === b.base)?.label || b.base;
      const about = [b.examples ? `Trained on ${b.examples} examples` : "Your brain", base ? `built on ${base}` : null, b.created].filter(Boolean).join(" · ");
      modelMenu.append(item(b.id, about, S.choice.brain === b.id,
        () => { S.choice = { base: b.base, brain: b.id }; store.set("octopus.choice", S.choice); }));
    }
    modelMenu.append(h("div", { class: "menu-sep" }));
  }
  const groups = [["Small and fast", 0, 3.5], ["Balanced", 3.5, 6], ["Big and smart (needs a strong computer)", 6, 99]];
  for (const [title, lo, hi] of groups) {
    const list = S.cfg.bases.filter((b) => { const n = parseFloat(b.size) || 0; return n >= lo && n < hi; });
    if (!list.length) continue;
    modelMenu.append(h("div", { class: "menu-note" }, title));
    for (const b of list) {
      modelMenu.append(item(b.label, `${b.maker ? b.maker + ". " : ""}${b.blurb}`, !S.choice.brain && S.choice.base === b.id,
        () => { S.choice = { base: b.id, brain: null }; store.set("octopus.choice", S.choice); }, b.size));
    }
  }
  modelMenu.append(h("div", { class: "menu-sep" }), h("div", { class: "menu-note" }, `Running on ${S.cfg.device}`));
  modelMenu.hidden = false; modelBtn.setAttribute("aria-expanded", "true");
  // The menu opens upwards, so keep it below the top of the window wherever the composer sits.
  modelMenu.style.maxHeight = Math.max(220, Math.min(560, modelBtn.getBoundingClientRect().top - 24)) + "px";
  modelMenu.querySelector('[aria-selected="true"]')?.scrollIntoView({ block: "nearest" });
}

// ---------- mode ----------
function setMode(mode, persist = true) {
  S.mode = mode; if (persist) store.set("octopus.mode", mode);
  document.body.dataset.mode = mode;
  $(".model-picker").hidden = mode === "model" || mode === "image";
  $("#input").placeholder = {
    chat: "Message Octopus", code: "Describe what to build, or paste code and an error",
    model: "Describe a 3D model, or attach a picture of it", image: "Describe an image",
  }[mode];
  $("#file").accept = mode === "model" ? "image/png,image/jpeg,image/webp" : "";
  $(".attach").hidden = mode === "image";
  $$(".type-item").forEach((b) => b.classList.toggle("active", b.dataset.type === mode && S.view === "chat"));
  if ((mode === "model" || mode === "image") !== (S.attachments[0]?.kind === "image") && S.attachments.length) { S.attachments = []; renderAttachments(); }
  renderGenOpts();
  const c = current(); if (c && !c.messages.length) c.mode = mode;
  if (!current()?.messages.length) renderEmpty();
}

// ---------- chats ----------
function newChat() {
  if (S.view !== "chat") showChatView();
  const existing = S.chats.find((c) => !c.messages.length);
  const c = existing || { id: uid(), title: "New chat", mode: S.mode, messages: [], updated: Date.now() };
  if (!existing) S.chats.unshift(c);
  S.currentId = c.id;
  renderThread(); renderHistory();
  $("#input").focus();
}
function openChat(id) {
  if (S.view !== "chat") showChatView();
  S.currentId = id;
  const c = current(); if (c.mode !== S.mode) setMode(c.mode);
  renderThread(); renderHistory();
}

const SUGGEST = {
  model: [
    ["Game prop", "A wooden treasure chest with iron bands and a gold lock"],
    ["Character", "A cute cartoon robot with round eyes and antenna, standing"],
    ["Weapon", "A fantasy sword with a glowing blue crystal in the hilt"],
    ["Vehicle", "A low-poly red sports car"],
  ],
  image: [
    ["Concept art", "A floating island city at sunset, fantasy concept art"],
    ["Game icon", "A shiny health potion bottle icon, game UI style, plain background"],
    ["Wallpaper", "A neon octopus swimming through a cyberpunk ocean, 4k wallpaper"],
    ["Photo", "A cosy desk setup with plants and warm lamp light, photo"],
  ],
  chat: [
    ["Explain something", "How do black holes evaporate?"],
    ["Plan my week", "Help me plan a productive week around school and the gym."],
    ["Write for me", "Write a short, friendly message inviting friends to a game night."],
    ["Brainstorm", "Give me 10 ideas for an iPhone app I could build."],
  ],
  code: [
    ["Write a script", "Write a Python script that renames every photo in a folder by its date taken."],
    ["Build a web page", "Make a single-file HTML page with a working calculator."],
    ["Explain code", "Explain what a Python decorator is with a simple example."],
    ["Solve a puzzle", "Write Python that finds all prime numbers below 1000 and prints how many there are."],
  ],
};
// The composer sits in the middle of a new chat's start screen and docks to the bottom once
// the conversation starts.
function dockComposer(inEmpty) {
  const form = $("#composer"); if (!form) return;
  if (inEmpty) $("#empty .slot")?.append(form);
  else if (form.parentElement !== $("#composer-wrap")) $("#composer-wrap").prepend(form);
  app.classList.toggle("is-empty", !!inEmpty);
}
function hideEmpty() { $("#empty").hidden = true; dockComposer(false); }

const fillPrompt = (p) => { const i = $("#input"); i.value = p; autosize(); updateSend(); i.focus(); };
const MODE_HERO = {
  chat: { eyebrow: null, sub: null },
  code: { eyebrow: "octopus ~ code", title: "What are we building?", sub: "Write, explain and fix code. HTML gets a live preview, and every block can be copied." },
  model: { eyebrow: "3D studio", title: "What should we model?", sub: "Describe it or attach a picture. You get a game-ready model: clean mesh, UVs, PBR textures and LODs." },
  image: { eyebrow: "Image studio", title: "What should we picture?", sub: "Concept art, icons, wallpapers and reference pictures for 3D models." },
};
const CODE_LANG = ["Python", "HTML", "Explain", "Python"];
const MODEL_STEPS = [["Describe or attach", "A sentence, or a picture of the object"], ["Sculpt", "Hunyuan3D makes the shape on a free cloud GPU"], ["Finish", "Octopus cleans, unwraps, bakes and builds LODs here"]];

function renderEmpty() {
  const empty = $("#empty"); const mode = S.mode;
  const hr = new Date().getHours();
  const hero = MODE_HERO[mode];
  const title = hero.title || (hr < 5 ? "Up late?" : hr < 12 ? "Good morning" : hr < 18 ? "Good afternoon" : "Good evening");
  const studioMode = mode === "model" || mode === "image";
  const ready = studioMode ? studioReady(mode) : true;
  const below = h("div", { class: "below" });

  if (mode === "chat") {
    below.append(h("div", { class: "pills" }, SUGGEST.chat.map(([t, p]) =>
      h("button", { class: "pill", onclick: () => fillPrompt(p) }, t))));
  } else if (mode === "code") {
    below.append(h("h6", {}, "Start from"), h("div", { class: "starter-grid" }, SUGGEST.code.map(([t, p], i) =>
      h("button", { class: "starter", onclick: () => fillPrompt(p) }, h("span", { class: "lang" }, CODE_LANG[i]), h("b", {}, t), h("span", {}, p)))),
      h("div", { class: "caps" }, [["Copy", ICON.copy], ["Live HTML preview", ICON.play], [S.cfg?.allowRun ? "Run on this computer" : "Run (turn on in Settings)", ICON.wrench]]
        .map(([t, i]) => h("span", {}, svg(i), t))));
  } else if (mode === "model") {
    below.append(h("div", { class: "steps" }, MODEL_STEPS.map(([t, d], i) => h("div", { class: "step-card" }, h("i", {}, String(i + 1)), h("b", {}, t), h("span", {}, d)))));
    if (ready) below.append(h("h6", {}, "Try one"), h("div", { class: "starter-grid four" }, SUGGEST.model.map(([t, p]) =>
      h("button", { class: "starter", onclick: () => fillPrompt(p) }, h("b", {}, t), h("span", {}, p)))));
    below.append(studioPanel(mode));
  } else {
    if (ready) below.append(h("div", { class: "pills" }, SUGGEST.image.map(([t, p]) => h("button", { class: "pill", title: p, onclick: () => fillPrompt(p) }, t))));
    below.append(studioPanel(mode));
  }

  dockComposer(false); // take the composer out before the start screen is rebuilt
  empty.replaceChildren(
    h("div", { class: "hero" },
      h("img", { class: "hero-octo", src: `/octo/${mode}.svg`, alt: "", "aria-hidden": "true" }),
      hero.eyebrow ? h("span", { class: "eyebrow" }, hero.eyebrow) : null,
      h("h1", { id: "greeting" }, title),
      hero.sub ? h("p", { class: "sub" }, hero.sub) : null),
    h("div", { class: "slot" }),
    below);
  empty.dataset.mode = mode;
  if (!empty.hidden) dockComposer(true);
  if (studioMode) renderGallery(mode);
  updateSend();
}

function renderThread() {
  const c = current(); const thread = $("#thread");
  thread.replaceChildren();
  $("#empty").hidden = !!c?.messages.length;
  if (!c?.messages.length) { renderEmpty(); dockComposer(true); return; }
  dockComposer(false);
  c.messages.forEach((m, i) => thread.append(m.role === "user" ? userMsg(m) : assistantMsg(m, i)));
  scrollDown(true);
}

function userMsg(m) {
  return h("div", { class: "msg user" }, h("div", { class: "bubble" },
    m.image ? h("img", { class: "attached", src: m.image.url, alt: "Attached picture" }) : null,
    m.files?.length ? h("div", { class: "files" }, m.files.map((f) => h("span", { class: "file-chip" }, svg(ICON.file), h("span", {}, f)))) : null,
    m.display ?? m.content));
}

// Models such as DeepSeek R1 reason before they answer. Ollama sends that separately; other engines
// put it in <think>…</think> at the start of the reply.
function splitThinking(text) {
  const m = /^\s*<think>([\s\S]*?)(<\/think>|$)/.exec(text);
  if (!m) return [null, text];
  return [m[1].trim(), m[2] ? text.slice(m[0].length).replace(/^\s+/, "") : ""];
}
function thinkingBox(text, live) {
  const box = h("details", { class: "thinking", open: live || null },
    h("summary", {}, h("span", { class: "think-label" }, live ? "Thinking…" : "Show thinking"), svg(ICON.chev, "chev")),
    h("div", { class: "think-text" }, text || ""));
  return box;
}

function assistantMsg(m, index) {
  if (m.kind) return assetMsg(m, index);
  const body = h("div", { class: "body" });
  if (m.thinking) body.append(thinkingBox(m.thinking, false));
  const md = h("div", { class: "md" }); body.append(md);
  renderMarkdown(md, m.content);
  if (m.error) body.append(h("div", { class: "notice" }, m.error));
  body.append(msgActions(m, index));
  return h("div", { class: "msg assistant" }, h("div", { class: "avatar", "aria-hidden": "true" }), body);
}

function msgActions(m, index) {
  const copy = h("button", { title: "Copy", onclick: () => { navigator.clipboard.writeText(m.content); copy.replaceChildren(svg(ICON.check)); setTimeout(() => copy.replaceChildren(svg(ICON.copy)), 1200); } }, svg(ICON.copy));
  const retry = h("button", { title: "Regenerate", onclick: () => regenerate(index) }, svg(ICON.retry));
  const meta = m.meta ? h("span", { class: "meta" }, m.meta) : null;
  return h("div", { class: "msg-actions" }, copy, retry, meta);
}

// ---------- markdown ----------
const RUNNABLE = new Set(["python", "py", "javascript", "js", "bash", "sh", "shell", "powershell"]);
marked.use({
  gfm: true, breaks: false,
  renderer: {
    code(code, lang) {
      const language = (lang || "").trim().split(/\s/)[0].toLowerCase();
      const valid = language && hljs.getLanguage(language);
      const html = valid ? hljs.highlight(code, { language }).value : hljs.highlightAuto(code).value;
      const canRun = S.cfg?.allowRun && RUNNABLE.has(language) && S.cfg.languages.includes(language);
      return `<div class="code" data-lang="${language}"><div class="code-head"><span>${language || "code"}</span><span>` +
        (canRun ? `<button data-act="run">${ICON_HTML.play}Run</button>` : "") +
        (language === "html" ? `<button data-act="preview">${ICON_HTML.play}Preview</button>` : "") +
        `<button data-act="copy">${ICON_HTML.copy}Copy</button></span></div><pre><code class="hljs">${html}</code></pre></div>`;
    },
  },
});
const ICON_HTML = Object.fromEntries(Object.entries(ICON).map(([k, v]) => [k, `<svg viewBox="0 0 24 24">${v}</svg>`]));

function renderMarkdown(el, text, streaming = false) {
  // Close an unterminated code fence while streaming so it renders as code.
  let src = text;
  if (streaming && (src.match(/```/g) || []).length % 2) src += "\n```";
  el.innerHTML = DOMPurify.sanitize(marked.parse(src), { ADD_ATTR: ["data-act", "data-lang"] });
  el.querySelectorAll("a").forEach((a) => { a.target = "_blank"; a.rel = "noopener noreferrer"; });
  if (streaming) el.classList.add("cursor"); else el.classList.remove("cursor");
}

$("#thread").addEventListener("click", async (e) => {
  const btn = e.target.closest("button[data-act]"); if (!btn) return;
  const block = btn.closest(".code"); const code = block.querySelector("code").innerText;
  if (btn.dataset.act === "copy") {
    await navigator.clipboard.writeText(code);
    btn.innerHTML = `${ICON_HTML.check}Copied`; setTimeout(() => (btn.innerHTML = `${ICON_HTML.copy}Copy`), 1400);
  } else if (btn.dataset.act === "run") runBlock(block, code, btn);
  else if (btn.dataset.act === "preview") openPreview(code);
});

async function runBlock(block, code, btn) {
  const lang = block.dataset.lang;
  block.nextElementSibling?.classList.contains("run-out") && block.nextElementSibling.remove();
  const out = h("details", { class: "disclosure run-out", open: true },
    h("summary", {}, h("span", { class: "spinner" }), "Running on your computer…", svg(ICON.chev, "chev")));
  block.after(out); btn.disabled = true;
  try {
    const r = await api("/api/run", { body: { language: lang, code } });
    const status = r.ok ? h("span", { class: "tool-ok" }, `Ran in ${(r.ms / 1000).toFixed(1)}s`) : h("span", { class: "tool-err" }, `Exited with code ${r.code}`);
    const inner = h("div", { class: "inner" });
    if (r.stdout) inner.append(h("span", { class: "label" }, "Output"), h("div", { class: "code" }, h("pre", {}, r.stdout)));
    if (r.stderr) inner.append(h("span", { class: "label" }, "Errors"), h("div", { class: "code" }, h("pre", { style: "color:#ffb4b4" }, r.stderr)));
    if (!r.stdout && !r.stderr) inner.append("No output.");
    if (!r.ok) inner.append(h("div", { style: "margin-top:10px" }, h("button", { class: "btn-ghost glass-btn", onclick: () => {
      $("#input").value = `I ran your ${lang} code and got this error:\n\n\`\`\`\n${(r.stderr || r.stdout).slice(-3000)}\n\`\`\`\n\nPlease fix it.`;
      autosize(); send();
    } }, svg(ICON.wrench), "Ask Octopus to fix it")));
    out.replaceChildren(h("summary", {}, svg(r.ok ? ICON.check : ICON.x), status, svg(ICON.chev, "chev")), inner);
  } catch (err) {
    out.replaceChildren(h("summary", {}, svg(ICON.x), h("span", { class: "tool-err" }, err.message)));
  } finally { btn.disabled = false; }
}

// ---------- composer ----------
const input = $("#input"), sendBtn = $("#send");
function autosize() { input.style.height = "auto"; input.style.height = Math.min(input.scrollHeight, innerHeight * 0.4) + "px"; updateSend(); }
function updateSend() {
  const blocked = (S.mode === "model" || S.mode === "image") && S.cfg && !studioReady(S.mode);
  input.disabled = !!blocked;
  sendBtn.disabled = !S.streaming && (blocked || (!input.value.trim() && !S.attachments.length)); sendBtn.classList.toggle("stop", !!S.streaming);
  // The drifting background makes the glass re-blur every frame. While a model is thinking, hold it
  // still so a CPU-only chat model gets the processor (it ran 2.5x slower with the page animating).
  document.body.classList.toggle("busy", !!S.streaming); }
input.addEventListener("input", autosize);
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing && !isMobile()) { e.preventDefault(); if (!S.streaming) send(); }
});
$("#composer").addEventListener("submit", (e) => { e.preventDefault(); S.streaming ? stop() : send(); });
addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") { e.preventDefault(); newChat(); }
  if (e.key === "Escape" && S.streaming) stop();
});

$("#file").addEventListener("change", async (e) => {
  if (S.mode === "model") { const f = e.target.files[0]; e.target.value = ""; if (f) attachImage(f); return; }
  for (const f of e.target.files) {
    if (f.size > 400_000) { toast(`${f.name} is too big (max 400 KB)`); continue; }
    const text = await f.text();
    if (/\u0000/.test(text.slice(0, 2000))) { toast(`${f.name} isn't a text file`); continue; }
    S.attachments.push({ name: f.name, text });
  }
  e.target.value = ""; renderAttachments();
});
function renderAttachments() {
  $("#attachments").replaceChildren(...S.attachments.map((a, i) => h("span", { class: "file-chip" + (a.kind === "image" ? " thumb-chip" : "") },
    a.kind === "image" ? h("img", { src: a.url, alt: "" }) : svg(ICON.file), h("span", {}, a.name),
    h("button", { type: "button", title: "Remove", onclick: () => { S.attachments.splice(i, 1); renderAttachments(); } }, svg(ICON.x)))));
  updateSend();
}

const thread = $("#thread"), scroller = $("#scroller");
let stick = true;
scroller.addEventListener("scroll", () => { stick = scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight < 80; });
function scrollDown(force) { if (force || stick) scroller.scrollTop = scroller.scrollHeight; }

async function send() {
  const text = input.value.trim();
  if ((!text && !S.attachments.length) || S.streaming) return;
  const c = current();
  if (S.mode === "model" || S.mode === "image") return sendStudio(c, text);
  const files = S.attachments.map((a) => a.name);
  const content = [text, ...S.attachments.map((a) => `\n\nFile: ${a.name}\n\`\`\`${extLang(a.name)}\n${a.text}\n\`\`\``)].join("").trim();
  c.messages.push({ role: "user", content, display: text || "(attached files)", files });
  if (c.title === "New chat") c.title = (text || files[0]).replace(/\s+/g, " ").slice(0, 48);
  c.mode = S.mode; c.updated = Date.now();
  input.value = ""; S.attachments = []; renderAttachments(); autosize();
  saveChats(); renderHistory();
  hideEmpty();
  thread.append(userMsg(c.messages.at(-1)));
  await generate(c);
}

function extLang(name) {
  const ext = name.split(".").pop().toLowerCase();
  return { py: "python", js: "javascript", ts: "typescript", sh: "bash", md: "markdown", rb: "ruby", rs: "rust", swift: "swift" }[ext] || ext;
}

async function regenerate(index) {
  const c = current(); if (S.streaming) return;
  c.messages = c.messages.slice(0, index);
  saveChats(); renderThread(); await generate(c);
}

async function generate(c) {
  const id = uid();
  const md = h("div", { class: "md" }, h("span", { class: "typing" }, h("i"), h("i"), h("i")));
  const status = h("div", { class: "notice info", hidden: true });
  const body = h("div", { class: "body" }, status, md);
  const el = h("div", { class: "msg assistant" }, h("div", { class: "avatar thinking", "aria-hidden": "true" }), body);
  thread.append(el); stick = true; scrollDown(true);

  const controller = new AbortController();
  S.streaming = { id, controller }; updateSend();
  let text = "", thinking = "", error = null, meta = null, raf = 0, last = 0, thinkBox = null;
  // Re-rendering Markdown and the glass effects on every token can starve a CPU-only chat model of
  // processor time on modest laptops, so repaint at most about 8 times a second while streaming.
  const paint = () => {
    const wait = 120 - (performance.now() - last);
    if (wait > 0) { raf = setTimeout(() => requestAnimationFrame(paint), wait); return; }
    raf = 0; last = performance.now();
    const [tagged, answer] = splitThinking(text);
    const thought = thinking || tagged;
    if (thought) {
      if (!thinkBox) { thinkBox = thinkingBox("", true); body.insertBefore(thinkBox, md); }
      thinkBox.querySelector(".think-text").textContent = thought;
      if (answer) thinkBox.querySelector(".think-label").textContent = "Thought it through";
    }
    if (answer || !thought) renderMarkdown(md, answer, true); else md.replaceChildren();
    scrollDown();
  };

  try {
    const res = await fetch("/api/chat", {
      method: "POST", headers: { "Content-Type": "application/json" }, signal: controller.signal, credentials: "same-origin",
      body: JSON.stringify({ id, mode: c.mode, base: S.choice.base, brain: S.choice.brain,
        messages: c.messages.map(({ role, content }) => ({ role, content })) }),
    });
    if (!res.ok) { const j = await res.json().catch(() => ({})); throw new Error(j.detail || failText(res.status)); }
    for await (const ev of sse(res)) {
      if (ev.t === "status") { status.hidden = false; status.replaceChildren(h("span", { class: "spinner", style: "display:inline-block;vertical-align:-2px;margin-right:8px" }), ev.d); }
      else if (ev.t === "engine") { status.hidden = true; }
      else if (ev.t === "text") { status.hidden = true; text += ev.d; if (!raf) raf = requestAnimationFrame(paint); }
      else if (ev.t === "think") { status.hidden = true; thinking += ev.d; if (!raf) raf = requestAnimationFrame(paint); }
      else if (ev.t === "error") error = ev.d;
      else if (ev.t === "done") meta = `${choiceLabel()} · ${(ev.ms / 1000).toFixed(1)}s`;
    }
  } catch (err) {
    if (err.name !== "AbortError") error = NET_ERR.test(err.message)
      ? "Lost the connection to Octopus. Check that its window (start.bat) is still open, then press Try again." : err.message;
  }
  clearTimeout(raf); cancelAnimationFrame(raf);
  S.streaming = null; updateSend();
  const [tagged, answer] = splitThinking(text);
  const msg = { role: "assistant", content: answer, meta };
  if (thinking || tagged) msg.thinking = (thinking || tagged).trim();
  if (error) msg.error = error;
  if (text || thinking || error) { c.messages.push(msg); c.updated = Date.now(); saveChats(); renderHistory(); }
  el.replaceWith(assistantMsg(msg, c.messages.length - 1));
  if (!text && !thinking && !error) el.remove();
  scrollDown();
}

async function stop() {
  if (!S.streaming) return;
  if (S.streaming.studio) { S.streaming.controller.abort(); return; }
  const { id, controller } = S.streaming;
  try { await api("/api/chat/stop", { body: { id } }); } catch {}
  setTimeout(() => controller.abort(), 400);
}

// ======================================================================
// Train section
// ======================================================================
const T = { datasets: [], selected: new Set(store.get("octopus.trainSel", [])), pairs: [], es: null, losses: [], lastSeq: 0 };

function showChatView() {
  S.view = "chat"; app.classList.remove("train-view");
  $("#train").hidden = true; $("#thread").hidden = false; $("#composer-wrap").hidden = false;
  $("#nav-train").classList.remove("active");
  if (location.hash) history.replaceState(null, "", "/");
  $$(".type-item").forEach((b) => b.classList.toggle("active", b.dataset.type === S.mode));
  renderThread();
}

async function openTrain() {
  S.view = "train"; app.classList.add("train-view");
  $$(".type-item").forEach((b) => b.classList.remove("active"));
  $("#thread").hidden = true; hideEmpty(); $("#composer-wrap").hidden = true;
  $("#nav-train").classList.add("active");
  history.replaceState(null, "", "#train");
  renderHistory();
  const root = $("#train"); root.hidden = false;
  root.replaceChildren(trainLayout());
  await refreshDatasets();
  const st = await api("/api/train/state").catch(() => null);
  if (st && st.status && st.status !== "idle") { applyState(st); listenTraining(0); }
  renderBrains();
}

function trainLayout() {
  const bases = S.cfg.bases.map((b) => h("option", { value: b.id, selected: b.id === (S.choice.base || S.cfg.defaultBase) }, `${b.label} (${b.size}, ${b.maker})`));
  const presets = { quick: [1, 8, 1000], balanced: [2, 16, 2000], deep: [3, 32, 5000] };
  const preset = store.get("octopus.preset", "balanced");

  const left = h("div", { class: "panel glass" },
    h("h2", {}, h("span", { class: "step" }, "1"), "Teach it"),
    h("p", { class: "hint" }, "Upload prompts and answers, write your own, or tick free open datasets. Octopus learns the style and knowledge in them."),
    h("label", { class: "dropzone", id: "drop" },
      svg(ICON.upload), h("b", {}, "Drop files here or click to upload"),
      h("span", {}, ".jsonl, .json, .csv or .txt with prompts and responses"),
      h("input", { type: "file", id: "ds-file", multiple: true, accept: ".jsonl,.json,.csv,.txt", hidden: true })),
    h("div", { class: "dataset-list", id: "ds-list" }),
    h("details", { class: "disclosure" },
      h("summary", {}, svg(ICON.spark), "Write examples yourself", h("span", { class: "pair-count", id: "pair-count" }), svg(ICON.chev, "chev")),
      h("div", { class: "inner", style: "white-space:normal;display:grid;gap:10px" },
        h("div", { class: "field" }, h("label", { for: "p-prompt" }, "When someone says…"), h("textarea", { id: "p-prompt", placeholder: "e.g. Who made you?" })),
        h("div", { class: "field" }, h("label", { for: "p-resp" }, "Octopus should answer…"), h("textarea", { id: "p-resp", placeholder: "e.g. Duncan built me. I'm Octopus, his own AI." })),
        h("div", { class: "row" },
          h("button", { class: "btn-ghost glass-btn", type: "button", onclick: addPair }, "Add example"),
          h("button", { class: "btn-ghost glass-btn", type: "button", onclick: savePairs }, "Save examples to a dataset")))),
    h("details", { class: "disclosure" },
      h("summary", {}, svg(ICON.file), "Supported file formats", svg(ICON.chev, "chev")),
      h("div", { class: "inner" },
`JSONL, one example per line:
{"prompt": "Hi", "response": "Hello! How can I help?"}
{"messages": [{"role": "user", "content": "…"}, {"role": "assistant", "content": "…"}]}

CSV with columns prompt,response (or instruction,output / question,answer)

Plain text, examples separated by a line with ---
User: What's 2+2?
Assistant: 4.`)));

  const settings = h("div", { class: "panel glass" },
    h("h2", {}, h("span", { class: "step" }, "2"), "Settings"),
    h("div", { class: "field" }, h("label", { for: "t-name" }, "Name this brain"), h("input", { type: "text", id: "t-name", value: store.get("octopus.trainName", "octopus-v1") })),
    h("div", { class: "field" }, h("label", { for: "t-base" }, "Start from"), h("select", { id: "t-base" }, bases)),
    h("div", { class: "field" }, h("label", {}, "How hard to train"),
      h("div", { class: "segmented glass-btn", id: "t-preset", style: "width:max-content" },
        ...Object.keys(presets).map((k) => h("button", { type: "button", "data-p": k, "aria-selected": String(k === preset),
          onclick: (e) => { store.set("octopus.preset", k); $$("#t-preset button").forEach((b) => b.setAttribute("aria-selected", String(b === e.currentTarget))); applyPreset(k); } },
          { quick: "Quick", balanced: "Balanced", deep: "Thorough" }[k])))),
    h("details", { class: "disclosure" },
      h("summary", {}, "Advanced", svg(ICON.chev, "chev")),
      h("div", { class: "inner", style: "white-space:normal;display:grid;gap:10px" },
        h("div", { class: "row" },
          numField("t-epochs", "Epochs", 2, 0.25, 0.25), numField("t-lr", "Learning rate", 0.0002, 0.00001, 0.00001),
          numField("t-rank", "LoRA rank", 16, 4, 4)),
        h("div", { class: "row" },
          numField("t-batch", "Batch size", 4, 1, 1), numField("t-accum", "Grad accumulation", 4, 1, 1),
          numField("t-len", "Max tokens per example", 1024, 128, 128)),
        numField("t-limit", "Examples to use from each open dataset", 2000, 100, 100))),
    h("div", { class: "field" }, h("label", { for: "t-test" }, "Test prompt after training"), h("input", { type: "text", id: "t-test", value: "Introduce yourself in two sentences." })),
    S.cfg.localGpu ? null : cloudTrainPanel(),
    h("div", { class: "row", hidden: !S.cfg.localGpu },
      h("button", { class: "btn-primary", id: "t-start", type: "button", onclick: startTraining }, svg(ICON.brain), "Start training"),
      h("button", { class: "btn-ghost glass-btn", id: "t-stop", type: "button", hidden: true, onclick: () => api("/api/train/stop", { body: {} }) }, "Stop and save")));

  const progress = h("div", { class: "panel glass", hidden: !S.cfg.localGpu },
    h("h2", {}, h("span", { class: "step" }, "3"), "Progress"),
    h("div", { class: "stats" },
      stat("s-step", "Step", "–"), stat("s-loss", "Loss", "–"), stat("s-eta", "Time left", "–")),
    h("div", { class: "progress" }, h("i", { id: "s-bar" })),
    h("canvas", { class: "chart", id: "s-chart" }),
    h("pre", { class: "log", id: "s-log" }, "Training will show up here. Lower loss means Octopus is fitting your examples better."),
    h("div", { id: "s-result" }));

  const brains = h("div", { class: "panel glass" },
    h("h2", {}, svg(ICON.brain), "Your brains"),
    h("p", { class: "hint" }, "Each training run saves a brain. Pick one in the model menu at the top to chat with it."),
    h("div", { class: "dataset-list", id: "brain-list" }));

  const wrap = h("div", {},
    h("div", { class: "train-hero" },
      h("img", { class: "train-octo", src: "/octo/train.svg", alt: "", "aria-hidden": "true" }),
      h("div", { class: "grow" }, h("h1", {}, "Train Octopus"), h("p", {}, S.cfg.localGpu ? `Fine-tune Octopus on your own examples, right on your GPU. Running on ${S.cfg.device}.` : "Fine-tune Octopus on your own examples using a free cloud GPU, then chat with it here.")),
      h("span", { class: "status-pill glass-btn", id: "t-pill" }, h("span", { class: "dot" }), h("span", { id: "t-pill-text" }, "Ready"))),
    h("div", { class: "grid-2" }, h("div", { style: "display:grid;gap:16px;align-content:start" }, left, settings), h("div", { style: "display:grid;gap:16px;align-content:start" }, progress, brains)));

  queueMicrotask(() => {
    wireDrop(); applyPreset(preset, true);
    const pre = $$("#t-preset button"); const sel = pre.find((b) => b.dataset.p === preset);
    requestAnimationFrame(() => positionThumb($("#t-preset"), sel));
    pre.forEach((b) => b.addEventListener("click", () => positionThumb($("#t-preset"), b)));
    updatePairCount();
  });
  return wrap;

  function applyPreset(k, init) {
    const [ep, rank, limit] = presets[k];
    if (!init || !$("#t-epochs").dataset.touched) { $("#t-epochs").value = ep; $("#t-rank").value = rank; $("#t-limit").value = limit; }
  }
}

function cloudTrainPanel() {
  const fileIn = h("input", { type: "file", accept: ".gguf,.zip", hidden: true, onchange: async (e) => {
    const f = e.target.files[0]; e.target.value = ""; if (!f) return;
    const name = ($("#t-name")?.value || f.name.replace(/\.[^.]+$/, "")).trim();
    toast(`Importing ${f.name}… big files take a minute`);
    try { const r = await uploadRaw("brain", f, name); toast(`Brain "${r.brain}" imported`); refreshBrainsCfg(); }
    catch (err) { toast(err.message); }
  } });
  return h("div", { class: "cloud", style: "padding:0;width:auto" },
    h("p", {}, "This computer doesn't have an NVIDIA GPU, so training happens free in the cloud:"),
    h("ol", {},
      h("li", {}, "Tick your datasets above and download them as one file."),
      h("li", {}, "Open the cloud notebook, upload that file, pick open datasets and press Run all."),
      h("li", {}, "Download the .gguf brain it makes and import it here. It appears in the model menu.")),
    h("div", { class: "row" },
      h("button", { class: "btn-ghost glass-btn", type: "button", onclick: exportData }, svg(ICON.download), "Download my data"),
      h("a", { class: "btn-primary", href: S.cfg.colabUrl, target: "_blank", rel: "noopener" }, svg(ICON.cloud), "Open cloud notebook"),
      h("label", { class: "btn-ghost glass-btn", style: "cursor:pointer" }, svg(ICON.upload), "Import brain", fileIn)));
}

async function exportData() {
  const sel = [...T.selected].filter((id) => id.startsWith("file:"));
  if (!sel.length) return toast("Tick at least one of your own datasets (open ones are picked in the notebook)");
  const res = await fetch("/api/datasets/export", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ datasets: sel }), credentials: "same-origin" });
  if (!res.ok) return toast((await res.json().catch(() => ({}))).detail || "Export failed");
  const a = h("a", { href: URL.createObjectURL(await res.blob()), download: "octopus-training.jsonl" }); a.click();
}

function positionThumb(segEl, btn) {
  let th = segEl.querySelector(".seg-thumb");
  if (!th) { th = h("span", { class: "seg-thumb" }); segEl.prepend(th); }
  th.style.width = btn.offsetWidth + "px"; th.style.transform = `translateX(${btn.offsetLeft - 3}px)`;
}
const numField = (id, label, value, step, min) => h("div", { class: "field" }, h("label", { for: id }, label),
  h("input", { type: "number", id, value, step, min, oninput: (e) => (e.target.dataset.touched = "1") }));
const stat = (id, label, value) => h("div", { class: "stat" }, h("small", {}, label), h("b", { id }, value));

function wireDrop() {
  const drop = $("#drop"), fileInput = $("#ds-file");
  const handle = async (files) => {
    for (const f of files) {
      if (f.size > 50_000_000) { toast(`${f.name} is over 50 MB`); continue; }
      try {
        const ds = await api("/api/datasets/upload", { body: { filename: f.name, text: await f.text() } });
        T.selected.add(ds.id); toast(`Added ${ds.count} examples from ${f.name}`);
      } catch (err) { toast(err.message); }
    }
    persistSel(); refreshDatasets();
  };
  fileInput.addEventListener("change", (e) => { handle(e.target.files); e.target.value = ""; });
  drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("drag"); });
  drop.addEventListener("dragleave", () => drop.classList.remove("drag"));
  drop.addEventListener("drop", (e) => { e.preventDefault(); drop.classList.remove("drag"); handle(e.dataTransfer.files); });
}
const persistSel = () => store.set("octopus.trainSel", [...T.selected]);

async function refreshDatasets() {
  T.datasets = await api("/api/datasets").catch(() => []);
  const list = $("#ds-list"); if (!list) return;
  list.replaceChildren(...T.datasets.map((d) => h("label", { class: "dataset" },
    h("input", { type: "checkbox", checked: T.selected.has(d.id), onchange: (e) => { e.target.checked ? T.selected.add(d.id) : T.selected.delete(d.id); persistSel(); } }),
    h("span", { class: "meta" }, h("b", {}, d.label), h("small", {}, d.kind === "file" ? `Your data · updated ${d.modified}` : `${d.blurb} ${d.license}.`)),
    d.kind === "file" ? h("span", { class: "count" }, `${d.count} examples`) : h("span", { class: "count" }, "Open dataset"),
    d.kind === "file" ? h("button", { class: "icon-btn", type: "button", title: "Delete dataset", onclick: async (e) => {
      e.preventDefault(); await api(`/api/datasets/${encodeURIComponent(d.id)}`, { method: "DELETE" }); T.selected.delete(d.id); persistSel(); refreshDatasets();
    } }, svg(ICON.trash)) : null)));
}

function addPair() {
  const p = $("#p-prompt").value.trim(), r = $("#p-resp").value.trim();
  if (!p || !r) return toast("Fill in both boxes");
  T.pairs.push({ prompt: p, response: r }); $("#p-prompt").value = ""; $("#p-resp").value = ""; $("#p-prompt").focus();
  updatePairCount(); toast("Example added. Save them when you're done.");
}
const updatePairCount = () => { const el = $("#pair-count"); if (el) el.textContent = T.pairs.length ? `${T.pairs.length} unsaved` : ""; };
async function savePairs() {
  if (!T.pairs.length) addPair();
  if (!T.pairs.length) return;
  try {
    const ds = await api("/api/datasets/pairs", { body: { pairs: T.pairs, name: "My examples" } });
    T.pairs = []; updatePairCount(); T.selected.add(ds.id); persistSel(); refreshDatasets(); toast(`Saved. "${ds.label}" now has ${ds.count} examples.`);
  } catch (err) { toast(err.message); }
}

async function startTraining() {
  const sel = [...T.selected].filter((id) => T.datasets.some((d) => d.id === id));
  if (!sel.length) return toast("Tick at least one dataset first");
  const name = $("#t-name").value.trim() || "octopus-v1"; store.set("octopus.trainName", name);
  const body = {
    name, base: $("#t-base").value, datasets: sel,
    epochs: +$("#t-epochs").value, learning_rate: +$("#t-lr").value, lora_rank: +$("#t-rank").value,
    batch_size: +$("#t-batch").value, grad_accum: +$("#t-accum").value, max_len: +$("#t-len").value,
    builtin_limit: +$("#t-limit").value, test_prompt: $("#t-test").value,
  };
  try {
    await api("/api/train/start", { body });
    T.losses = []; $("#s-log").textContent = ""; $("#s-result").replaceChildren();
    applyState({ status: "starting" }); listenTraining(0);
  } catch (err) { toast(err.message); }
}

function listenTraining(after) {
  T.es?.close();
  const es = new EventSource(`/api/train/events?after=${after}`); T.es = es;
  es.onmessage = (m) => {
    const ev = JSON.parse(m.data); T.lastSeq = ev.seq + 1;
    if (ev.t === "log") { const log = $("#s-log"); if (log) { log.textContent += (log.textContent ? "\n" : "") + ev.msg; log.scrollTop = log.scrollHeight; } }
    else if (ev.t === "progress") { T.losses.push(ev.smooth); applyState({ status: "training", ...ev }); }
    else if (ev.t === "status") { applyState(ev); if (["done", "error", "stopped"].includes(ev.status)) finishTraining(ev); }
  };
  es.onerror = () => { es.close(); setTimeout(async () => { const st = await api("/api/train/state").catch(() => null); if (st?.running) listenTraining(T.lastSeq); }, 2000); };
}

function applyState(st) {
  const pill = $("#t-pill"), text = $("#t-pill-text"); if (!pill) return;
  const running = ["starting", "preparing", "training"].includes(st.status);
  pill.classList.toggle("running", running);
  text.textContent = { starting: "Starting…", preparing: "Preparing data and model…", training: "Training", done: "Finished", error: "Error", stopped: "Stopped", idle: "Ready" }[st.status] || st.status;
  $("#t-start").disabled = running; $("#t-stop").hidden = !running;
  $("#train-badge").hidden = !running;
  if (st.step != null) {
    $("#s-step").textContent = `${st.step} / ${st.total_steps}`;
    $("#s-loss").textContent = (st.smooth ?? st.loss)?.toFixed(3);
    $("#s-eta").textContent = fmtTime(st.eta);
    $("#s-bar").style.width = `${(100 * st.step) / st.total_steps}%`;
    drawChart();
  }
}

function finishTraining(ev) {
  T.es?.close(); refreshBrainsCfg();
  const box = $("#s-result"); if (!box) return;
  if (ev.status === "error") { box.replaceChildren(h("div", { class: "notice" }, ev.error)); return; }
  if (!ev.brain) { box.replaceChildren(h("div", { class: "notice info" }, ev.msg || "Stopped.")); return; }
  box.replaceChildren(h("div", { class: "notice info", style: "display:grid;gap:10px" },
    h("b", {}, `🎉 Brain "${ev.brain}" is ready`),
    ev.sample ? h("div", {}, h("small", { class: "muted" }, "Its answer to your test prompt:"), h("div", { style: "white-space:pre-wrap;margin-top:4px" }, ev.sample)) : null,
    h("div", {}, h("button", { class: "btn-primary", onclick: () => { S.choice = { base: $("#t-base").value, brain: ev.brain }; store.set("octopus.choice", S.choice); renderModelButton(); newChat(); } }, "Chat with it"))));
}

async function refreshBrainsCfg() { S.cfg.brains = await api("/api/brains").catch(() => S.cfg.brains); renderBrains(); }
function renderBrains() {
  const list = $("#brain-list"); if (!list) return;
  if (!S.cfg.brains.length) { list.replaceChildren(h("div", { class: "history-empty" }, "No brains yet. Train one and it appears here.")); return; }
  list.replaceChildren(...S.cfg.brains.map((b) => h("div", { class: "dataset" },
    svg(ICON.brain),
    h("span", { class: "meta" }, h("b", {}, b.id), h("small", {}, `${b.examples ?? "?"} examples · loss ${b.final_loss ?? "?"} · ${b.created ?? ""}`)),
    h("button", { class: "btn-ghost glass-btn", onclick: () => { S.choice = { base: b.base, brain: b.id }; store.set("octopus.choice", S.choice); renderModelButton(); newChat(); } }, "Use"),
    h("button", { class: "icon-btn", title: "Delete brain", onclick: async () => {
      if (!confirmDelete(b.id)) return;
      await api(`/api/brains/${encodeURIComponent(b.id)}`, { method: "DELETE" });
      if (S.choice.brain === b.id) { S.choice.brain = null; store.set("octopus.choice", S.choice); renderModelButton(); }
      refreshBrainsCfg();
    } }, svg(ICON.trash)))));
}
const pendingDelete = new Set();
function confirmDelete(id) {
  if (pendingDelete.has(id)) { pendingDelete.delete(id); return true; }
  pendingDelete.add(id); toast(`Press delete again to remove "${id}"`); setTimeout(() => pendingDelete.delete(id), 3000); return false;
}

function drawChart() {
  const c = $("#s-chart"); if (!c) return;
  const dpr = devicePixelRatio || 1, w = c.clientWidth, hgt = c.clientHeight;
  c.width = w * dpr; c.height = hgt * dpr;
  const ctx = c.getContext("2d"); ctx.scale(dpr, dpr);
  const data = T.losses; if (data.length < 2) return;
  const max = Math.max(...data), min = Math.min(...data), pad = 14;
  const x = (i) => pad + (i / (data.length - 1)) * (w - 2 * pad);
  const y = (v) => pad + (1 - (v - min) / (max - min || 1)) * (hgt - 2 * pad);
  const accent = getComputedStyle(document.documentElement).getPropertyValue("--accent").trim();
  const grad = ctx.createLinearGradient(0, 0, 0, hgt);
  grad.addColorStop(0, accent + "55"); grad.addColorStop(1, accent + "00");
  ctx.beginPath(); data.forEach((v, i) => (i ? ctx.lineTo(x(i), y(v)) : ctx.moveTo(x(i), y(v))));
  ctx.lineWidth = 2; ctx.strokeStyle = accent; ctx.lineJoin = "round"; ctx.stroke();
  ctx.lineTo(x(data.length - 1), hgt - pad); ctx.lineTo(x(0), hgt - pad); ctx.closePath(); ctx.fillStyle = grad; ctx.fill();
  ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue("--text-3"); ctx.font = "11px system-ui";
  ctx.fillText(max.toFixed(2), pad, pad + 4); ctx.fillText(min.toFixed(2), pad, hgt - pad - 4);
}
const fmtTime = (s) => (s == null ? "–" : s < 60 ? `${s}s` : s < 3600 ? `${Math.round(s / 60)}m` : `${(s / 3600).toFixed(1)}h`);


// ======================================================================
// Studio: Modelling (3D) and Image chats
// ======================================================================
ICON.cube = '<path d="M12 3 4 7.5v9L12 21l8-4.5v-9Z"/><path d="m4 7.5 8 4.5 8-4.5M12 12v9"/>';
ICON.download = '<path d="M12 4v12M7 11l5 5 5-5M5 20h14"/>';
ICON.cloud = '<path d="M7 18.5h10a4 4 0 0 0 .6-8A6 6 0 0 0 6 9.6a4.5 4.5 0 0 0 1 8.9Z"/>';
ICON.code = '<path d="m8.5 7.5-5 4.5 5 4.5M15.5 7.5l5 4.5-5 4.5"/>';
ICON.image = '<rect x="3.5" y="4.5" width="17" height="15" rx="3"/><circle cx="9" cy="10" r="1.8"/><path d="m4 17 5-4.5 4 3.5 3-2.5 4 3.5"/>';
ICON_HTML.cube = `<svg viewBox="0 0 24 24">${ICON.cube}</svg>`;

const studioReady = (mode) => !!(mode === "model" ? S.cfg?.studio?.model3d : S.cfg?.studio?.image);
const runsLocally = (mode) => {
  const st = S.cfg?.studio; if (!st) return false;
  const where = S.settings?.studioWhere || "auto"; const local = st.local?.[mode === "model" ? "model3d" : "image"];
  return S.cfg.mock || (local && where !== "cloud") || (!st.cloud && local);
};
const GEN = Object.assign({ preset: "quality", texture: true, budget: "game", texSize: 2048, height: 1.0, lods: true, imageModel: "sdxl", aspect: "1:1", count: 1 },
  store.get("octopus.gen", {}));
if (GEN.v !== 2) Object.assign(GEN, { v: 2, preset: "quality", budget: "game", texSize: 2048, height: 1.0, lods: true });
const saveGen = () => store.set("octopus.gen", GEN);

function renderGenOpts() {
  const box = $("#gen-opts"); if (!box) return;
  box.replaceChildren();
  const studio = S.cfg?.studio;
  if (S.mode === "model") {
    const pick = (key, title, options, cls = "opt", num = false) => h("select", { class: cls, title,
      onchange: (e) => { GEN[key] = num ? +e.target.value : e.target.value; saveGen(); } },
      options.map(([v, label]) => h("option", { value: v, selected: String(v) === String(GEN[key]) }, label)));
    const toggle = (key, label, title, cls = "opt") => h("button", { type: "button", class: cls, title, "aria-pressed": String(GEN[key]),
      onclick: (e) => { GEN[key] = !GEN[key]; saveGen(); e.currentTarget.setAttribute("aria-pressed", String(GEN[key])); } }, label);
    const height = h("label", { class: "opt opt-num opt-hide-m", title: "Real-world height in metres. The model is scaled to this, pivot at the bottom centre." },
      h("input", { type: "number", min: "0.05", max: "500", step: "any", value: GEN.height, "aria-label": "Height in metres",
        onchange: (e) => { GEN.height = Math.max(0.05, +e.target.value || 1); e.target.value = GEN.height; saveGen(); } }), "m");
    box.append(
      pick("preset", "Detail: Quality takes longer and gives sharper shapes", [["quality", "Quality"], ["fast", "Fast"]]),
      pick("budget", "Triangle budget for the game mesh (LOD0)", (studio?.budgets || []).map((b) => [b.id, b.label.split(" · ")[0] + " " + (b.tris / 1000) + "k"])),
      pick("texSize", "PBR texture resolution", (studio?.textureSizes || [1024, 2048, 4096]).map((n) => [n, n / 1024 + "K"]), "opt opt-hide-m", true),
      height,
      toggle("texture", "Textures", "Paint colours and materials onto the model"),
      toggle("lods", "LODs", "Also make LOD1 and LOD2, lighter versions for when the model is far away", "opt opt-hide-m"));
  } else if (S.mode === "image") {
    box.append(...[
      runsLocally("image") ? h("select", { class: "opt", title: "Image model", onchange: (e) => { GEN.imageModel = e.target.value; saveGen(); } },
        (studio?.imageModels || []).map((m) => h("option", { value: m.id, selected: m.id === GEN.imageModel }, m.label.split(" · ")[0]))) : null,
      h("select", { class: "opt", title: "Shape", onchange: (e) => { GEN.aspect = e.target.value; saveGen(); } },
        ["1:1", "16:9", "9:16", "4:3"].map((a) => h("option", { value: a, selected: a === GEN.aspect }, a))),
      h("select", { class: "opt opt-hide-m", title: "How many", onchange: (e) => { GEN.count = +e.target.value; saveGen(); } },
        [1, 2, 4].map((n) => h("option", { value: n, selected: n === GEN.count }, `${n} image${n > 1 ? "s" : ""}`)))].filter(Boolean));
  }
}

async function attachImage(file) {
  if (file.size > 20_000_000) return toast("That picture is over 20 MB");
  const data = await new Promise((res, rej) => { const r = new FileReader(); r.onload = () => res(r.result); r.onerror = rej; r.readAsDataURL(file); });
  try {
    const meta = await api("/api/upload-image", { body: { name: file.name, data } });
    S.attachments = [{ kind: "image", name: file.name, id: meta.id, url: `/files/${meta.id}/${meta.files.png}` }];
    renderAttachments();
  } catch (err) { toast(err.message); }
}

async function sendStudio(c, text) {
  const img = S.attachments.find((a) => a.kind === "image");
  if (S.mode === "model" && !text && !img) return;
  const user = { role: "user", content: text || "Make a 3D model from this picture", display: text || "Make a 3D model from this picture" };
  if (img) user.image = { id: img.id, url: img.url, imageName: img.imageName };
  c.messages.push(user);
  if (c.title === "New chat") c.title = (text || img?.name || "3D model").replace(/\s+/g, " ").slice(0, 48);
  c.mode = S.mode; c.updated = Date.now();
  input.value = ""; S.attachments = []; renderAttachments(); autosize(); saveChats(); renderHistory();
  hideEmpty(); thread.append(userMsg(user));
  await generateAsset(c, S.mode, text, img?.id, img?.imageName);
}

async function generateAsset(c, kind, prompt, imageId, imageName) {
  const label = h("span", {}, "Starting…"), pct = h("span", {}, ""), bar = h("i");
  const prog = h("div", { class: "gen-progress" },
    h("img", { class: "work-octo", src: `/octo/${kind === "model" ? "model" : "image"}.svg`, alt: "", "aria-hidden": "true" }),
    h("div", { class: "grow" }, h("div", { class: "row" }, h("span", { style: "display:flex;gap:8px;align-items:center" }, h("span", { class: "spinner" }), label), pct),
      h("div", { class: "progress" }, bar)));
  const el = h("div", { class: "msg assistant" }, h("div", { class: "avatar thinking", "aria-hidden": "true" }), h("div", { class: "body" }, prog));
  thread.append(el); stick = true; scrollDown(true);
  const controller = new AbortController();
  S.streaming = { id: uid(), controller, studio: true }; updateSend();
  let asset = null, error = null; const started = Date.now();
  const url = kind === "model" ? "/api/generate/model" : "/api/generate/image";
  const body = kind === "model"
    ? { prompt, image_id: imageId, image_name: imageName, preset: GEN.preset, texture: GEN.texture, ...finishOpts() }
    : { prompt, model: GEN.imageModel, aspect: GEN.aspect, count: GEN.count };
  try {
    const res = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body), signal: controller.signal, credentials: "same-origin" });
    if (!res.ok) { const j = await res.json().catch(() => ({})); throw new Error(j.detail || failText(res.status)); }
    for await (const ev of sse(res)) {
      if (ev.t === "progress") { label.textContent = ev.stage; if (ev.pct != null) { bar.style.width = ev.pct + "%"; pct.textContent = Math.round(ev.pct) + "%"; } }
      else if (ev.t === "done") asset = ev.asset;
      else if (ev.t === "error") error = ev.d;
    }
  } catch (err) {
    if (err.name === "AbortError") error = "Stopped. The current step finishes in the background.";
    else if (NET_ERR.test(err.message) || err instanceof TypeError) {
      label.textContent = "Lost the connection to Octopus, checking…";
      const r = await recoverStudio(label);
      asset = r.asset || null; error = r.error || null;
    } else error = err.message;
  }
  S.streaming = null; updateSend();
  const msg = { role: "assistant", kind, content: asset ? `[${kind === "model" ? "3D model" : "image"}: ${prompt || "from picture"}]` : "", asset, error,
    meta: asset ? `${((Date.now() - started) / 1000).toFixed(0)}s` : null };
  c.messages.push(msg); c.updated = Date.now(); saveChats(); renderHistory();
  el.replaceWith(assetMsg(msg, c.messages.length - 1)); scrollDown();
}

const finishOpts = () => ({ budget: GEN.budget, texture_size: GEN.texSize, height: GEN.height, lods: GEN.lods });
const fileUrl = (a, name) => `/files/${a.id}/${name}` + (a.updated && /\.(glb|png)$/.test(name) && name !== a.reference ? `?v=${a.updated}` : "");
const dl = (href, label) => h("a", { class: "btn-ghost glass-btn", href, download: "" }, svg(ICON.download), label);

function assetMsg(m, index) {
  const body = h("div", { class: "body" });
  const a = m.asset;
  if (m.error && !a) body.append(h("div", { class: "notice" }, m.error));
  if (a && m.kind === "model") {
    const glb = fileUrl(a, a.files.glb);
    const viewer = document.createElement("model-viewer");
    Object.entries({ src: glb, alt: a.prompt || "3D model", "camera-controls": "", "auto-rotate": "", "shadow-intensity": "1",
      exposure: "1.05", "environment-image": "neutral", "touch-action": "pan-y", "interaction-prompt": "none" }).forEach(([k, v]) => viewer.setAttribute(k, v));
    const st = a.stats || {};
    const size = st.size_m ? `${st.size_m[1].toFixed(2)} m tall` : "";
    const facts = [st.faces ? `${st.faces.toLocaleString()} triangles` : "", a.textured === false ? "Untextured" : st.pbr ? `${st.texture / 1024}K PBR` : (a.textured ? "Textured" : "Untextured"),
      st.lods?.length > 1 ? `${st.lods.length} LODs` : "", size].filter(Boolean);
    const card = h("div", { class: "asset" }, viewer, h("div", { class: "asset-bar" },
      a.reference ? h("img", { class: "ref-thumb", src: fileUrl(a, a.reference), alt: "Reference" }) : null,
      h("span", { class: "meta", title: [a.engine, st.source_faces ? `made from a ${st.source_faces.toLocaleString()} triangle sculpt` : ""].filter(Boolean).join(", ") },
        [...facts, m.meta].filter(Boolean).join(" · ")),
      a.files.pack ? h("a", { class: "btn-primary", style: "height:32px;font-size:13px;padding:0 12px", href: fileUrl(a, a.files.pack), download: "",
        title: "GLB, OBJ and STL for every LOD, PBR textures and a how-to for Unity, Unreal, Godot and Blender" }, svg(ICON.download), "Game pack") : null,
      dl(glb, "GLB"), a.files.obj ? dl(fileUrl(a, a.files.obj), "OBJ") : null, a.files.stl ? dl(fileUrl(a, a.files.stl), "STL") : null,
      a.sources ? h("button", { class: "btn-ghost glass-btn", title: "Rebuild the game files with the triangle budget, texture size and height chosen below (no new generation)",
        onclick: (e) => rebuildAsset(m, index, e.currentTarget) }, svg(ICON.retry), "Rebuild") : null,
      h("button", { class: "btn-ghost glass-btn", onclick: () => useInCode(a) }, svg(ICON.code), "Use in Code")));
    body.append(card);
    if (a.note) body.append(h("div", { class: "notice" }, a.note));
  } else if (a && m.kind === "image") {
    const imgs = a.images || [a.files.png];
    body.append(h("div", { class: "asset" }, h("div", { class: "img-grid" + (imgs.length === 1 ? " one" : "") }, imgs.map((name) => h("div", { class: "img-cell" },
      h("img", { src: fileUrl(a, name), alt: a.prompt || "Generated image", loading: "lazy" }),
      h("div", { class: "over" }, dl(fileUrl(a, name), "Save"),
        h("button", { class: "btn-ghost glass-btn", onclick: () => modelFromImage(a, name) }, svg(ICON.cube), "Make 3D"))))),
      h("div", { class: "asset-bar" }, h("span", { class: "meta" }, [IMAGE_LABEL(a.model), a.aspect, m.meta].filter(Boolean).join(" · ")))));
  }
  const retry = h("button", { title: "Try again", onclick: () => regenerateAsset(index) }, svg(ICON.retry));
  body.append(h("div", { class: "msg-actions" }, retry, m.meta ? h("span", { class: "meta" }, m.meta) : null));
  return h("div", { class: "msg assistant" }, h("div", { class: "avatar", "aria-hidden": "true" }), body);
}
const IMAGE_LABEL = (id) => S.cfg?.studio?.imageModels?.find((m) => m.id === id)?.label.split(" · ")[1];

async function rebuildAsset(m, index, btn) {
  const c = current(); if (S.streaming) return;
  btn.disabled = true; btn.lastChild.textContent = "Rebuilding…";
  try {
    const a = await api(`/api/assets/${m.asset.id}/refinish`, { body: finishOpts() });
    m.asset = a; saveChats();
    const el = btn.closest(".msg"); el.replaceWith(assetMsg(m, index));
    toast(`Rebuilt: ${a.stats.faces.toLocaleString()} triangles, ${a.stats.texture / 1024}K textures`);
  } catch (err) { toast(err.message); btn.disabled = false; btn.lastChild.textContent = "Rebuild"; }
}

async function regenerateAsset(index) {
  const c = current(); if (S.streaming) return;
  const user = c.messages[index - 1]; c.messages = c.messages.slice(0, index); saveChats(); renderThread();
  await generateAsset(c, c.mode, user.display === "Make a 3D model from this picture" ? "" : user.display, user.image?.id, user.image?.imageName);
}

function modelFromImage(a, name) {
  if (!studioReady("model")) return toast("3D modelling isn't set up. Run start again to finish installing.");
  setMode("model"); newChat();
  S.attachments = [{ kind: "image", name: a.prompt?.slice(0, 30) || "image", id: a.id, url: fileUrl(a, name), imageName: name }];
  renderAttachments(); $("#input").value = a.prompt || ""; autosize();
  toast("Picture attached. Press send to make it 3D.");
}

function useInCode(a) {
  const url = location.origin + fileUrl(a, a.files.glb);
  setMode("code"); newChat();
  $("#input").value =
`I made a 3D model called "${a.prompt || "my model"}". It's a GLB file at:
${url}

Write a single self-contained HTML file using Three.js from a CDN (import map for three and three/addons) that:
- loads this model with GLTFLoader, centres it on a grassy ground plane and scales it to about 2 units tall
- adds soft lighting, shadows and a sky colour
- lets me walk around it in third person with WASD and look around with the mouse, like a simple game
Use a \`\`\`html code block.`;
  autosize(); input.focus();
  toast("Ready in Code. Press send, then Preview the HTML it writes.");
}

function openPreview(code) {
  const m = $("#preview"); m.hidden = false;
  $("#preview-frame").srcdoc = code;
}
$("#preview-close").addEventListener("click", () => { $("#preview").hidden = true; $("#preview-frame").srcdoc = ""; });
$("#preview").addEventListener("click", (e) => { if (e.target.id === "preview") $("#preview-close").click(); });

// ---------- gallery & cloud panel ----------
async function renderGallery(mode) {
  const items = await api(`/api/assets?kind=${mode}`).catch(() => []);
  if (S.mode !== mode || current()?.messages.length) return;
  $("#empty").querySelector(".gallery")?.remove();
  if (!items.length) return;
  ($("#empty .below") || $("#empty")).append(h("div", { class: "gallery" }, h("h6", {}, mode === "model" ? "Your 3D models" : "Your images"),
    h("div", { class: "gallery-grid" }, items.slice(0, 12).map((a) => h("button", { class: "gallery-item", title: a.prompt || "",
      onclick: () => openAssetChat(a, mode) },
      mode === "image" ? h("img", { src: fileUrl(a, (a.images || [a.files.png])[0]), alt: "", loading: "lazy" })
        : a.reference ? h("img", { src: fileUrl(a, a.reference), alt: "", loading: "lazy" }) : h("div", { class: "icon3d" }, svg(ICON.cube)),
      h("span", {}, a.prompt || "Untitled"))))));
}

function openAssetChat(a, mode) {
  const c = current();
  c.messages.push({ role: "user", content: a.prompt || "Imported model", display: a.prompt || "Imported model" },
    { role: "assistant", kind: mode, asset: a, content: `[${mode}: ${a.prompt}]` });
  c.title = (a.prompt || "Imported model").slice(0, 48); c.mode = mode; c.updated = Date.now();
  saveChats(); renderThread(); renderHistory();
}

function studioPanel(mode) {
  const fileIn = h("input", { type: "file", accept: ".glb,.obj,.stl,.zip", hidden: true, onchange: (e) => { const f = e.target.files[0]; e.target.value = ""; if (f) importModel(f); } });
  const local = runsLocally(mode);
  const st = S.cfg.studio || {};
  const ready = studioReady(mode);
  const where = S.cfg.mock ? "Preview mode: placeholder results, no models needed."
    : !ready ? "The cloud helper isn't installed yet. Close Octopus and run start again."
    : local ? `Runs on your ${S.cfg.device || "GPU"}.`
    : mode === "model" ? `Runs free on Hugging Face's cloud GPUs with ${st.cloud3d?.[0] || "Hunyuan3D"}, then Octopus finishes the mesh on this computer.`
    : "Runs free on Hugging Face's cloud GPUs with FLUX.1 schnell.";
  const quota = !local && !S.cfg.mock && !S.settings?.hasToken && S.cfg.user?.role === "owner"
    ? h("button", { class: "linkish", onclick: openSettings }, "Add a free Hugging Face token for more daily generations") : null;
  return h("div", { class: "cloud glass studio-panel" },
    h("p", {}, svg(local ? ICON.cube : ICON.cloud), h("span", {}, where, " ", quota)),
    mode === "model" ? h("div", { class: "row" },
      h("span", { class: "muted", style: "font-size:13px;margin-right:auto" }, "Every model comes game-ready: cleaned, UV-unwrapped, PBR-baked with LODs."),
      h("label", { class: "btn-ghost glass-btn", style: "cursor:pointer;height:32px;font-size:13px" }, svg(ICON.upload), "Import a 3D model", fileIn)) : null);
}

// ---------- settings ----------
async function loadSettings() { S.settings = await api("/api/settings").catch(() => ({})); }

function openSettings() {
  $("#settings-modal")?.remove();
  const s = S.settings || {};
  const u = S.cfg.user || {};
  const owner = u.role === "owner";
  const close = () => modal.remove();

  // --- your account ---
  const nameIn = h("input", { type: "text", value: u.name || "", autocomplete: "name" });
  const oldPw = h("input", { type: "password", placeholder: "Current password", autocomplete: "current-password" });
  const newPw = h("input", { type: "password", placeholder: "New password (6 or more characters)", autocomplete: "new-password" });
  const accErr = h("p", { class: "error" });
  const saveAccount = async () => {
    accErr.textContent = "";
    try {
      const body = { name: nameIn.value };
      if (newPw.value) Object.assign(body, { old_password: oldPw.value, new_password: newPw.value });
      const r = await api("/api/me", { body });
      S.cfg.user = r.user; renderAccount(); oldPw.value = newPw.value = "";
      toast(body.new_password ? "Password changed. Other devices were signed out." : "Account saved");
    } catch (e) { accErr.textContent = e.message; }
  };
  const accountSection = h("section", { class: "set-section" },
    h("h4", {}, "Your account"),
    h("div", { class: "acct-head" }, h("span", { class: "initials big" }, $("#account-initials").textContent),
      h("div", {}, h("b", {}, u.name), h("small", { class: "muted" }, `@${u.username} · ${owner ? "Owner" : "Member"}`))),
    h("div", { class: "field" }, h("label", {}, "Name"), nameIn),
    h("div", { class: "field" }, h("label", {}, "Change password"), oldPw, newPw),
    accErr,
    h("div", { class: "row", style: "justify-content:flex-end" }, h("button", { class: "btn-ghost glass-btn", onclick: saveAccount }, "Save account")));

  // --- people (owner only) ---
  const people = h("div", { class: "people" });
  const signups = h("input", { type: "checkbox" });
  const renderPeople = (d) => {
    signups.checked = d.signupsOpen;
    people.replaceChildren(...d.users.map((x) => h("div", { class: "person" },
      h("span", { class: "initials" }, x.name.split(/\s+/).map((w) => w[0]).join("").slice(0, 2).toUpperCase()),
      h("div", { class: "grow" }, h("b", {}, x.name), h("small", { class: "muted" }, `@${x.username}${x.role === "owner" ? " · Owner" : ""}`)),
      x.id !== u.id && x.role !== "owner" ? h("button", { class: "btn-ghost glass-btn danger", onclick: async (e) => {
        const b = e.currentTarget;
        if (b.dataset.sure !== "1") { b.dataset.sure = "1"; b.textContent = "Remove? Click again"; return; }
        try { await api(`/api/accounts/${x.id}`, { method: "DELETE" }); renderPeople(await api("/api/accounts")); toast(`Removed ${x.name}`); }
        catch (err) { toast(err.message); }
      } }, "Remove") : null)));
  };
  signups.addEventListener("change", async () => {
    try { renderPeople(await api("/api/accounts", { body: { signupsOpen: signups.checked } })); } catch (e) { toast(e.message); }
  });
  const peopleSection = owner ? h("section", { class: "set-section" },
    h("h4", {}, "People on this Octopus"),
    h("label", { class: "check-row" }, signups, h("span", {}, "Let others create accounts from the sign-in screen")),
    people) : null;
  if (owner) api("/api/accounts").then(renderPeople).catch(() => {});

  // --- studio (owner only) ---
  const tok = h("input", { type: "password", placeholder: s.hasToken ? `Saved (${s.tokenHint}). Paste a new one to replace it` : "hf_…", autocomplete: "off", spellcheck: "false" });
  const whereSel = h("select", {}, [["auto", "Automatic"], ["local", "This computer (NVIDIA GPU)"], ["cloud", "Free cloud (Hugging Face)"]]
    .map(([v, l]) => h("option", { value: v, selected: v === (s.studioWhere || "auto") }, l)));
  const err = h("p", { class: "error" });
  const save = async (clear = false) => {
    err.textContent = "";
    try {
      const body = { studio_where: whereSel.value };
      if (clear) body.hf_token = ""; else if (tok.value.trim()) body.hf_token = tok.value.trim();
      S.settings = await api("/api/settings", { body });
      toast(clear ? "Token removed" : "Settings saved"); setMode(S.mode, false);
    } catch (e) { err.textContent = e.message; }
  };
  const studioSection = owner ? h("section", { class: "set-section" },
    h("h4", {}, "3D and images"),
    h("div", { class: "field" }, h("label", {}, "Where to make images and 3D models"), whereSel,
      h("p", { class: "muted" }, S.cfg.studio?.local?.model3d ? "Your NVIDIA GPU is set up for 3D." : "This computer has no NVIDIA GPU set up for 3D, so Automatic uses the free cloud.")),
    h("div", { class: "field" }, h("label", {}, "Hugging Face token (free, optional)"), tok,
      h("p", { class: "muted" }, "The free cloud gives everyone a small daily allowance of GPU time. A free Hugging Face account gets more. ",
        h("a", { href: "https://huggingface.co/join", target: "_blank", rel: "noopener" }, "Create an account"), ", then make a ",
        h("a", { href: "https://huggingface.co/settings/tokens/new?tokenType=read", target: "_blank", rel: "noopener" }, "Read token"),
        " and paste it here. It stays on this computer and is shared by everyone on this Octopus.")),
    err,
    h("div", { class: "row", style: "justify-content:flex-end" },
      s.hasToken && !s.tokenFromEnv ? h("button", { class: "btn-ghost glass-btn", onclick: () => save(true) }, "Remove token") : null,
      h("button", { class: "btn-primary", onclick: () => save() }, "Save"))) : null;

  const modal = h("div", { id: "settings-modal", class: "modal", onclick: (e) => { if (e.target === modal) close(); } },
    h("div", { class: "modal-card glass settings-card", role: "dialog", "aria-label": "Settings" },
      h("div", { class: "modal-head" }, h("b", {}, "Settings"),
        h("button", { class: "icon-btn", "aria-label": "Close settings", onclick: close, style: "margin-left:auto" }, svg('<path d="M7 7l10 10M17 7 7 17"/>'))),
      h("div", { class: "settings-body" }, accountSection, peopleSection, studioSection)));
  document.body.append(modal); nameIn.focus();
}
$("#settings-btn")?.addEventListener("click", () => { openSettings(); });

async function uploadRaw(kind, file, name = "") {
  const res = await fetch(`/api/import?kind=${kind}&filename=${encodeURIComponent(file.name)}&name=${encodeURIComponent(name)}`, {
    method: "POST", headers: { "Content-Type": "application/octet-stream" }, body: file, credentials: "same-origin" });
  const j = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(j.detail || `Import failed (${res.status})`);
  return j;
}

async function importModel(file) {
  toast(`Importing ${file.name}…`);
  try {
    const a = await uploadRaw("model", file, file.name.replace(/\.[^.]+$/, ""));
    if (S.mode !== "model") setMode("model");
    openAssetChat(a, "model");
  } catch (err) { toast(err.message); }
}

// ---------- Ollama status banner ----------
function ollamaBanner() {
  if (S.cfg?.backend !== "ollama" || S.cfg.ollamaRunning) return;
  const b = h("div", { class: "notice banner" }, "Octopus chats through Ollama, which isn't running. ",
    h("a", { href: "https://ollama.com/download", target: "_blank", rel: "noopener" }, "Install Ollama"),
    ", open it, then refresh this page.");
  $("#scroller").prepend(b);
}

boot();
