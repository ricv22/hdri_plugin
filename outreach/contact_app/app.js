const state = {
  contacts: [],
  filtered: [],
  selected: null,
  language: "en",
  priority: "all",
  query: "",
  openingIndex: 0,
  sendConfigured: false,
  sendFrom: "",
  reached: {},
  reachView: "all",
  renderedCompany: "",
  fileMode: false,
};

const elements = {
  count: document.querySelector("#contact-count"),
  list: document.querySelector("#contact-list"),
  search: document.querySelector("#search"),
  empty: document.querySelector("#empty-state"),
  composer: document.querySelector("#composer"),
  template: document.querySelector("#contact-template"),
  companyName: document.querySelector("#company-name"),
  companyLocation: document.querySelector("#company-location"),
  companyPriority: document.querySelector("#company-priority"),
  companyFocus: document.querySelector("#company-focus"),
  companyEmail: document.querySelector("#company-email"),
  companyWebsite: document.querySelector("#company-website"),
  companySignal: document.querySelector("#company-signal"),
  companyHook: document.querySelector("#company-hook"),
  openings: document.querySelector("#opening-options"),
  opening: document.querySelector("#opening"),
  langCz: document.querySelector("#lang-cz"),
  langEn: document.querySelector("#lang-en"),
  preview: document.querySelector("#message-preview"),
  messageTo: document.querySelector("#message-to"),
  messageSubject: document.querySelector("#message-subject"),
  meta: document.querySelector("#message-meta"),
  resetMessage: document.querySelector("#reset-message"),
  copyOpening: document.querySelector("#copy-opening"),
  copyMessage: document.querySelector("#copy-message"),
  sendMessage: document.querySelector("#send-message"),
  copyStatus: document.querySelector("#copy-status"),
  reachedCount: document.querySelector("#reached-count"),
  companyReached: document.querySelector("#company-reached"),
  companyReplied: document.querySelector("#company-replied"),
  responseMeta: document.querySelector("#response-meta"),
  responseBody: document.querySelector("#response-body"),
  fetchResponse: document.querySelector("#fetch-response"),
  saveResponse: document.querySelector("#save-response"),
  deleteContact: document.querySelector("#delete-contact"),
  companyCard: document.querySelector(".company-card"),
  sessionLabel: document.querySelector("#session-label"),
  loginOpen: document.querySelector("#login-open"),
  logout: document.querySelector("#logout"),
  loginOverlay: document.querySelector("#login-overlay"),
  loginForm: document.querySelector("#login-form"),
  loginUser: document.querySelector("#login-user"),
  loginPassword: document.querySelector("#login-password"),
  loginRemember: document.querySelector("#login-remember"),
  loginStatus: document.querySelector("#login-status"),
  loginSkip: document.querySelector("#login-skip"),
  loginSubmit: document.querySelector("#login-submit"),
  modeBanner: document.querySelector("#mode-banner"),
};

const REACHED_STORAGE_KEY = "outreach-reached";
const RESPONSE_STORAGE_KEY = "outreach-responses";
const DELETED_STORAGE_KEY = "outreach-deleted";

function apiAvailable() {
  return location.protocol === "http:" || location.protocol === "https:";
}

function loadLocalReached() {
  try {
    const raw = JSON.parse(localStorage.getItem(REACHED_STORAGE_KEY) || "[]");
    return Array.isArray(raw) ? raw : [];
  } catch {
    return [];
  }
}

function saveLocalReached(rows) {
  localStorage.setItem(REACHED_STORAGE_KEY, JSON.stringify(rows));
}

function loadLocalResponses() {
  try {
    const raw = JSON.parse(localStorage.getItem(RESPONSE_STORAGE_KEY) || "{}");
    return raw && typeof raw === "object" ? raw : {};
  } catch {
    return {};
  }
}

function saveLocalResponses(records) {
  localStorage.setItem(RESPONSE_STORAGE_KEY, JSON.stringify(records));
}

function loadDeletedNames() {
  try {
    const raw = JSON.parse(localStorage.getItem(DELETED_STORAGE_KEY) || "[]");
    return new Set(Array.isArray(raw) ? raw : []);
  } catch {
    return new Set();
  }
}

function rememberDeleted(company) {
  const names = loadDeletedNames();
  names.add(company);
  localStorage.setItem(DELETED_STORAGE_KEY, JSON.stringify([...names]));
}

function applyLocalResponses(contacts) {
  const saved = loadLocalResponses();
  for (const contact of contacts) {
    const reply = saved[contact.company];
    if (!reply) continue;
    contact.response_body = reply.body || "";
    contact.response_from = reply.from || "";
    contact.response_subject = reply.subject || "";
    contact.response_date = reply.date || "";
    contact.response_source = reply.source || "manual";
    if (reply.body || reply.status === "replied") contact.status = "replied";
  }
}

function isReplied(contact) {
  return contact?.status === "replied" || Boolean(contact?.response_body);
}

const fixedCopy = {
  en: {
    subject: "Freelance 3D / VFX support",
    greeting: (company) => `Hello ${company} team,`,
    body: `I'm a 3D artist, and I help productions on a freelance basis when they need a specific shot covered, or just some extra capacity.

I can take on a full shot, or jump in on just the part of post you're working on, from rotoscope, camera tracking, and matchmove through modeling and animation to compositing, editing, and the final render.

Work samples:
https://richardandrys.com

If you're on something like this right now, I'm available.

Thank you for your time,
Richard Andrýs
rich.andrys@gmail.com`,
  },
  cz: {
    subject: "Externí spolupráce – 3D grafik / VFX",
    greeting: () => "Dobrý den,",
    body: `Jsem 3D grafik a produkcím pomáhám externě, když potřebují pokrýt konkrétní záběr nebo jen doplnit kapacitu.

Shot umím převzít celý, stejně tak se můžu zapojit jen do části postprodukce, kterou právě řešíte, od rotoscope, camera trackingu a matchmove přes tvorbu a animaci modelů až po compositing, střih a finální render.

Ukázky mé práce:
https://richardandrys.com

Pokud právě něco podobného řešíte, jsem k dispozici.

Děkuji za váš čas,
Richard Andrýs
rich.andrys@gmail.com`,
  },
};

function strengths(contact) {
  const text = `${contact.focus} ${contact.why_fit}`.toLowerCase();
  if (text.includes("game") || text.includes("cinematic")) {
    return { en: "3D animation, cinematics and visual storytelling", cz: "3D animaci, cinematiku a vizuální storytelling" };
  }
  if (text.includes("product") || text.includes("industrial") || text.includes("luxury")) {
    return { en: "product CGI, animation and polished visual design", cz: "produktové CGI, animaci a precizní vizuální design" };
  }
  if (text.includes("film") || text.includes("series") || text.includes("television")) {
    return { en: "film VFX, CGI and post-production", cz: "filmové VFX, CGI a postprodukci" };
  }
  if (text.includes("motion") || text.includes("animation")) {
    return { en: "animation, motion design and CGI", cz: "animaci, motion design a CGI" };
  }
  if (text.includes("immersive") || text.includes("virtual production")) {
    return { en: "CGI, immersive work and virtual production", cz: "CGI, immersive tvorbu a virtuální produkci" };
  }
  return { en: "3D, VFX and post-production", cz: "3D, VFX a postprodukci" };
}

function openingSuggestions(contact, language) {
  const area = strengths(contact)[language];
  if (language === "cz") {
    return [
      `Narazil jsem na ${contact.company} a zaujal mě způsob, jakým propojujete ${area}.`,
      `Píšu vám, protože vaše práce v oblasti ${area} je blízko tomu, čemu se věnuji i já.`,
      `Vaše studio mě zaujalo rozsahem práce v oblasti ${area} a rád bych nabídl externí 3D/VFX podporu.`,
    ];
  }
  return [
    `I came across ${contact.company} and was struck by how your work brings together ${area}.`,
    `I'm reaching out because your work across ${area} overlaps closely with the kind of production support I offer.`,
    `Your studio caught my attention for its work in ${area}, and I'd like to offer freelance 3D/VFX support.`,
  ];
}

function defaultLanguage(contact) {
  return String(contact.outreach_language || "").toLowerCase().startsWith("czech") ? "cz" : "en";
}

function applyFilters() {
  const query = state.query.trim().toLowerCase();
  state.filtered = state.contacts.filter((contact) => {
    const rank = Number(contact.priority);
    const priorityMatch =
      state.priority === "all" ||
      (state.priority === "5" && rank === 5) ||
      (state.priority === "4" && rank >= 4);
    const replied = isReplied(contact);
    const reached = isReached(contact.company);
    const reachMatch =
      state.reachView === "all" ||
      (state.reachView === "reached" && reached) ||
      (state.reachView === "replied" && replied) ||
      (state.reachView === "open" && !reached);
    const haystack = `${contact.company} ${contact.country} ${contact.city} ${contact.focus}`.toLowerCase();
    return priorityMatch && reachMatch && (!query || haystack.includes(query));
  });

  if (!state.selected || !state.filtered.includes(state.selected)) {
    state.selected = state.filtered[0] || null;
  }
  renderList();
  renderSelected();
}

function isReached(company) {
  return Boolean(state.reached[company]);
}

function applyReached(payload) {
  state.reached = Object.fromEntries((payload.reached || []).map((row) => [row.company, row]));
  updateReachedCount();
  if (state.contacts.length) applyFilters();
}

function updateReachedCount() {
  const reached = Object.keys(state.reached).length;
  const total = state.contacts.length;
  elements.reachedCount.textContent = `Reached ${reached} / ${total}`;
}

function renderList() {
  elements.list.replaceChildren();
  elements.count.textContent = String(state.filtered.length);
  updateReachedCount();

  for (const contact of state.filtered) {
    const row = elements.template.content.firstElementChild.cloneNode(true);
    const reached = isReached(contact.company);
    row.querySelector(".contact-name").textContent = contact.company;
    row.querySelector(".contact-location").textContent = [contact.city, contact.country].filter(Boolean).join(" · ");
    row.querySelector(".contact-priority").textContent = contact.priority;
    row.classList.toggle("is-active", contact === state.selected);
    row.classList.toggle("is-reached", reached);
    row.classList.toggle("is-replied", isReplied(contact));
    row.setAttribute("aria-selected", String(contact === state.selected));
    const checkbox = row.querySelector(".contact-reached");
    checkbox.checked = reached;
    checkbox.addEventListener("click", (event) => event.stopPropagation());
    checkbox.addEventListener("change", () => {
      toggleReached(contact.company, checkbox.checked);
    });
    row.querySelector(".contact-select").addEventListener("click", () => selectContact(contact));
    elements.list.append(row);
  }
}

function selectContact(contact) {
  state.selected = contact;
  state.language = defaultLanguage(contact);
  state.openingIndex = 0;
  state.renderedCompany = "";
  renderList();
  renderSelected();
}

function renderSelected() {
  const contact = state.selected;
  const hasContact = Boolean(contact);
  elements.composer.hidden = !hasContact;
  elements.empty.hidden = hasContact;

  if (!contact) {
    state.renderedCompany = "";
    elements.empty.innerHTML = "<p>No contacts match this filter.</p>";
    return;
  }

  elements.companyName.textContent = contact.company;
  elements.companyLocation.textContent = [contact.city, contact.country].filter(Boolean).join(" · ");
  elements.companyPriority.textContent = `Priority ${contact.priority}`;
  elements.companyFocus.textContent = contact.focus;
  elements.companyEmail.textContent = contact.contact_email || "No public email";
  elements.companyEmail.href = contact.contact_email ? `mailto:${contact.contact_email}` : "#";
  elements.companyWebsite.href = contact.website;
  elements.companySignal.textContent = contact.signal;
  elements.companyHook.textContent = contact.personalization_hook;
  elements.companyReached.checked = isReached(contact.company);
  elements.companyReplied.checked = contact.status === "replied";
  elements.companyCard.classList.toggle("is-reached", isReached(contact.company));
  elements.companyCard.classList.toggle("is-replied", isReplied(contact));
  const replyFrom = contact.response_from || contact.contact_email || "";
  const replyWhen = contact.response_date ? ` · ${contact.response_date}` : "";
  const replySource = contact.response_source === "gmail" ? " · from Gmail" : "";
  elements.responseMeta.textContent = contact.response_body
    ? `${replyFrom}${replyWhen}${replySource}`
    : "No reply saved yet. Fetch also picks up a different address in the same conversation, or at the same company when the subject is your outreach email.";
  if (state.renderedCompany !== contact.company) {
    elements.responseBody.value = contact.response_body || "";
    state.renderedCompany = contact.company;
    renderMessage();
  }
}

function renderMessage() {
  const contact = state.selected;
  if (!contact) return;

  elements.langCz.classList.toggle("is-active", state.language === "cz");
  elements.langEn.classList.toggle("is-active", state.language === "en");
  const suggestions = openingSuggestions(contact, state.language);
  const selectedOpening = suggestions[state.openingIndex] || suggestions[0];

  elements.openings.replaceChildren();
  suggestions.forEach((text, index) => {
    const label = document.createElement("label");
    label.className = "opening-option";
    label.classList.toggle("is-active", index === state.openingIndex);
    const radio = document.createElement("input");
    radio.type = "radio";
    radio.name = "opening";
    radio.checked = index === state.openingIndex;
    const span = document.createElement("span");
    span.textContent = text;
    label.append(radio, span);
    label.addEventListener("click", () => {
      state.openingIndex = index;
      elements.opening.value = text;
      renderMessage();
    });
    elements.openings.append(label);
  });

  elements.opening.value = selectedOpening;
  updatePreview();
}

function fullMessage() {
  const contact = state.selected;
  if (!contact) return "";
  const copy = fixedCopy[state.language];
  const opening = elements.opening.value.trim();
  return `${copy.greeting(contact.company)}

${opening}

${copy.body}`;
}

function currentMessage() {
  return elements.preview.value;
}

function updatePreview() {
  if (!state.selected) return;
  elements.messageTo.value = state.selected.contact_email || "";
  elements.messageSubject.value = fixedCopy[state.language].subject;
  elements.preview.value = fullMessage();
  updateMeta();
}

function updateMeta() {
  const recipient = elements.messageTo.value.trim() || "No email";
  const subject = elements.messageSubject.value.trim() || "No subject";
  const sendHint = state.sendConfigured
    ? `Send as ${state.sendFrom}`
    : state.fileMode
      ? "Copy the message — sending needs Open Composer.command on your Mac"
      : "Log in to send, or copy the message";
  elements.meta.textContent = `${recipient} · ${subject} · ${sendHint}`;
  elements.sendMessage.disabled = false;
  elements.sendMessage.title = state.sendConfigured
    ? `Send this one email as ${state.sendFrom}`
    : state.fileMode
      ? "Copy the message; sending needs the local composer"
      : "Log in with Gmail to send, or copy the message";
  updateSessionUI();
}

function showCopyFeedback(button, message) {
  const original = button.dataset.originalLabel || button.textContent;
  button.dataset.originalLabel = original;
  button.textContent = state.language === "cz" ? "Zkopírováno ✓" : "Copied ✓";
  elements.copyStatus.textContent = message;
  window.clearTimeout(copyText.timeout);
  copyText.timeout = window.setTimeout(() => {
    button.textContent = original;
    elements.copyStatus.textContent = "";
  }, 2200);
}

async function copyText(text, successMessage, button) {
  // Give immediate feedback. Clipboard permission prompts can otherwise make
  // the click look unresponsive while the browser is waiting.
  showCopyFeedback(button, successMessage);
  try {
    await navigator.clipboard.writeText(text);
  } catch {
    const textarea = document.createElement("textarea");
    textarea.value = text;
    textarea.style.position = "fixed";
    textarea.style.opacity = "0";
    document.body.append(textarea);
    textarea.select();
    document.execCommand("copy");
    textarea.remove();
  }
}

function changeLanguage(language) {
  state.language = language;
  state.openingIndex = 0;
  renderMessage();
}

async function sendCurrentEmail() {
  const contact = state.selected;
  if (!contact) return;
  if (state.fileMode || !apiAvailable()) {
    elements.copyStatus.textContent =
      "Z tohoto souboru se maily neodesílají. Zkopíruj text, nebo na Macu spusť Open Composer.command.";
    return;
  }
  if (!state.sendConfigured) {
    showLogin();
    elements.copyStatus.textContent = "Nejdřív se přihlas Gmailem, nebo zkopíruj mail a pošli ho ručně.";
    return;
  }
  const recipient = elements.messageTo.value.trim();
  const subject = elements.messageSubject.value.trim();
  const body = currentMessage().trim();
  if (!recipient || !subject || !body) {
    elements.copyStatus.textContent = "Fill in To, Subject, and the email first.";
    return;
  }
  const confirmed = window.confirm(`Send this one email to ${recipient}?`);
  if (!confirmed) return;
  elements.sendMessage.disabled = true;
  try {
    const response = await fetch("/api/send", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        company: contact.company,
        to: recipient,
        subject,
        body,
        confirm: true,
      }),
    });
    const result = await response.json();
    if (!response.ok || !result.ok) {
      throw new Error(result.error || `HTTP ${response.status}`);
    }
    elements.copyStatus.textContent = `Sent to ${result.to}`;
    if (result.reached) applyReached(result.reached);
    else await toggleReached(contact.company, true);
  } catch (error) {
    elements.copyStatus.textContent = `Not sent: ${error.message}`;
  } finally {
    elements.sendMessage.disabled = false;
  }
}

function replaceContacts(contacts) {
  const selectedName = state.selected?.company;
  state.contacts = contacts;
  state.selected = contacts.find((contact) => contact.company === selectedName) || contacts[0] || null;
  state.renderedCompany = "";
  applyFilters();
}

async function deleteCurrentContact() {
  const contact = state.selected;
  if (!contact) return;
  const confirmed = window.confirm(`Delete ${contact.company} from the contact sheet?`);
  if (!confirmed) return;
  if (state.fileMode || !apiAvailable()) {
    rememberDeleted(contact.company);
    const responses = loadLocalResponses();
    delete responses[contact.company];
    saveLocalResponses(responses);
    state.contacts = state.contacts.filter((item) => item.company !== contact.company);
    state.selected = state.contacts[0] || null;
    state.renderedCompany = "";
    applyFilters();
    return;
  }
  try {
    const response = await fetch("/api/contact/delete", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ company: contact.company }),
    });
    const result = await response.json();
    if (!response.ok || !result.ok) throw new Error(result.error || `HTTP ${response.status}`);
    if (result.reached) applyReached(result);
    replaceContacts(result.contacts || []);
    elements.copyStatus.textContent = `${contact.company} deleted`;
  } catch (error) {
    elements.copyStatus.textContent = `Not deleted: ${error.message}`;
  }
}

async function setReplied(replied) {
  const contact = state.selected;
  if (!contact) return;
  if (state.fileMode || !apiAvailable()) {
    const responses = loadLocalResponses();
    const existing = responses[contact.company] || {};
    if (replied) {
      responses[contact.company] = { ...existing, status: "replied" };
    } else if (existing.body) {
      responses[contact.company] = { ...existing, status: "sent" };
    } else {
      delete responses[contact.company];
    }
    saveLocalResponses(responses);
    contact.status = replied ? "replied" : "sent";
    renderList();
    renderSelected();
    return;
  }
  try {
    const response = await fetch("/api/contact/replied", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ company: contact.company, replied }),
    });
    const result = await response.json();
    if (!response.ok || !result.ok) throw new Error(result.error || `HTTP ${response.status}`);
    if (result.reached) applyReached(result);
    replaceContacts(result.contacts || []);
  } catch (error) {
    elements.copyStatus.textContent = `Could not update reply: ${error.message}`;
    renderSelected();
  }
}

async function saveCurrentResponse() {
  const contact = state.selected;
  if (!contact) return;
  const body = elements.responseBody.value.trim();
  if (!body) {
    elements.copyStatus.textContent = "Paste their reply first.";
    return;
  }
  if (state.fileMode || !apiAvailable()) {
    const responses = loadLocalResponses();
    responses[contact.company] = {
      from: contact.contact_email || "",
      subject: "",
      date: new Date().toISOString().slice(0, 10),
      body,
      source: "manual",
    };
    saveLocalResponses(responses);
    contact.response_body = body;
    contact.response_from = contact.contact_email || "";
    contact.response_date = new Date().toISOString().slice(0, 10);
    contact.response_source = "manual";
    contact.status = "replied";
    state.renderedCompany = "";
    renderList();
    renderSelected();
    elements.copyStatus.textContent = "Reply saved on this computer";
    return;
  }
  elements.saveResponse.disabled = true;
  try {
    const response = await fetch("/api/contact/response", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ company: contact.company, body }),
    });
    const result = await response.json();
    if (!response.ok || !result.ok) throw new Error(result.error || `HTTP ${response.status}`);
    if (result.reached) applyReached(result);
    replaceContacts(result.contacts || []);
    elements.copyStatus.textContent = "Reply saved on this contact";
  } catch (error) {
    elements.copyStatus.textContent = `Reply not saved: ${error.message}`;
  } finally {
    elements.saveResponse.disabled = false;
  }
}

async function fetchCurrentResponse() {
  const contact = state.selected;
  if (!contact) return;
  if (state.fileMode || !apiAvailable()) {
    elements.copyStatus.textContent = "Fetching mail needs the composer running on your Mac.";
    return;
  }
  if (!state.sendConfigured) {
    showLogin();
    elements.copyStatus.textContent = "Log in with Gmail first, then fetch the reply.";
    return;
  }
  elements.fetchResponse.disabled = true;
  elements.copyStatus.textContent = "Looking in Gmail…";
  try {
    const response = await fetch("/api/contact/response", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ company: contact.company, fetch: true }),
    });
    const result = await response.json();
    if (!response.ok || !result.ok) throw new Error(result.error || `HTTP ${response.status}`);
    if (result.reached) applyReached(result);
    replaceContacts(result.contacts || []);
    elements.copyStatus.textContent = "Reply attached from Gmail";
  } catch (error) {
    elements.copyStatus.textContent = `No reply attached: ${error.message}`;
  } finally {
    elements.fetchResponse.disabled = false;
  }
}

async function toggleReached(company, reached) {
  if (state.fileMode || !apiAvailable()) {
    const records = Object.fromEntries(loadLocalReached().map((row) => [row.company, row]));
    if (reached) {
      records[company] = records[company] || {
        company,
        reached_on: new Date().toISOString().slice(0, 10),
        source: "manual",
      };
    } else {
      delete records[company];
    }
    const rows = Object.values(records);
    saveLocalReached(rows);
    applyReached({ reached: rows, count: rows.length, total: state.contacts.length });
    return;
  }
  try {
    const response = await fetch("/api/reached", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ company, reached, source: "manual" }),
    });
    const result = await response.json();
    if (!response.ok || !result.ok) throw new Error(result.error || `HTTP ${response.status}`);
    applyReached(result);
  } catch (error) {
    elements.copyStatus.textContent = `Could not update reached status: ${error.message}`;
    renderList();
    renderSelected();
  }
}

function applySession(status) {
  state.sendConfigured = Boolean(status.configured);
  state.sendFrom = status.from || "";
  updateSessionUI();
  updateMeta();
}

function updateSessionUI() {
  elements.sessionLabel.textContent = state.sendConfigured ? state.sendFrom : "Not logged in";
  elements.loginOpen.hidden = state.sendConfigured;
  elements.logout.hidden = !state.sendConfigured;
  if (elements.modeBanner) {
    elements.modeBanner.hidden = !state.fileMode;
    elements.modeBanner.textContent = "Copy mode — sending needs Open Composer.command on your Mac";
  }
  if (state.sendConfigured) {
    elements.loginOverlay.hidden = true;
    elements.loginPassword.value = "";
    elements.loginStatus.textContent = "";
  }
}

function showLogin() {
  elements.loginOverlay.hidden = false;
  elements.loginStatus.textContent = "";
  elements.loginStatus.classList.remove("is-error", "is-ok");
  elements.loginUser.focus();
}

async function submitLogin(event) {
  event.preventDefault();
  const user = elements.loginUser.value.trim();
  const password = elements.loginPassword.value;
  elements.loginSubmit.disabled = true;
  elements.loginStatus.classList.remove("is-error", "is-ok");
  elements.loginStatus.textContent = "Checking Gmail login…";
  try {
    const response = await fetch("/api/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        user,
        password,
        persist: elements.loginRemember.checked,
      }),
    });
    const result = await response.json();
    if (!response.ok || !result.ok) throw new Error(result.error || `HTTP ${response.status}`);
    applySession(result);
    elements.loginStatus.classList.add("is-ok");
    elements.loginStatus.textContent = `Logged in as ${result.from}`;
  } catch (error) {
    elements.loginStatus.classList.add("is-error");
    elements.loginStatus.textContent = error.message;
  } finally {
    elements.loginSubmit.disabled = false;
    elements.loginPassword.value = "";
  }
}

async function logout() {
  const response = await fetch("/api/logout", { method: "POST" });
  const result = response.ok ? await response.json() : { configured: false, from: "" };
  applySession(result);
}

async function init() {
  try {
    let contacts = [];
    if (apiAvailable()) {
      try {
        const [contactsResponse, statusResponse, reachedResponse] = await Promise.all([
          fetch("/api/contacts", { cache: "no-store" }),
          fetch("/api/send-status", { cache: "no-store" }),
          fetch("/api/reached", { cache: "no-store" }),
        ]);
        if (!contactsResponse.ok) throw new Error(`HTTP ${contactsResponse.status}`);
        const data = await contactsResponse.json();
        contacts = data.contacts || [];
        if (statusResponse.ok) {
          applySession(await statusResponse.json());
        }
        if (reachedResponse.ok) {
          applyReached(await reachedResponse.json());
        }
      } catch (apiError) {
        if (!window.EMBEDDED_CONTACTS?.contacts?.length) throw apiError;
        state.fileMode = true;
        contacts = window.EMBEDDED_CONTACTS.contacts.filter((contact) => !loadDeletedNames().has(contact.company));
        applyLocalResponses(contacts);
        applyReached({ reached: loadLocalReached() });
      }
    } else if (window.EMBEDDED_CONTACTS?.contacts?.length) {
      state.fileMode = true;
      contacts = window.EMBEDDED_CONTACTS.contacts.filter((contact) => !loadDeletedNames().has(contact.company));
      applyLocalResponses(contacts);
      applyReached({ reached: loadLocalReached() });
    } else {
      throw new Error("No contact list found.");
    }
    if (!contacts.length) throw new Error("Contact list is empty.");
    state.contacts = contacts;
    state.filtered = contacts;
    state.selected = contacts[0] || null;
    if (state.selected) state.language = defaultLanguage(state.selected);
    applyFilters();
    updateSessionUI();
  } catch (error) {
    elements.empty.hidden = false;
    elements.composer.hidden = true;
    elements.empty.innerHTML = `<p>Could not load contacts: ${error.message}</p>`;
  }
}

elements.search.addEventListener("input", (event) => {
  state.query = event.target.value;
  applyFilters();
});

document.querySelectorAll(".filter").forEach((button) => {
  button.addEventListener("click", () => {
    if (button.dataset.reach) {
      state.reachView = state.reachView === button.dataset.reach ? "all" : button.dataset.reach;
    } else {
      state.priority = button.dataset.priority;
    }
    document.querySelectorAll("[data-priority]").forEach((item) => {
      item.classList.toggle("is-active", item.dataset.priority === state.priority);
    });
    document.querySelectorAll("[data-reach]").forEach((item) => {
      item.classList.toggle("is-active", item.dataset.reach === state.reachView);
    });
    applyFilters();
  });
});

elements.langCz.addEventListener("click", () => changeLanguage("cz"));
elements.langEn.addEventListener("click", () => changeLanguage("en"));
elements.opening.addEventListener("input", updatePreview);
elements.messageTo.addEventListener("input", updateMeta);
elements.messageSubject.addEventListener("input", updateMeta);
elements.resetMessage.addEventListener("click", updatePreview);
elements.copyOpening.addEventListener("click", () =>
  copyText(
    elements.opening.value.trim(),
    state.language === "cz" ? "Úvod zkopírován" : "Opening copied",
    elements.copyOpening,
  ),
);
elements.copyMessage.addEventListener("click", () =>
  copyText(
    currentMessage(),
    state.language === "cz" ? "Celý e-mail zkopírován" : "Full message copied",
    elements.copyMessage,
  ),
);
elements.sendMessage.addEventListener("click", sendCurrentEmail);
elements.companyReached.addEventListener("change", () => {
  if (state.selected) toggleReached(state.selected.company, elements.companyReached.checked);
});
elements.companyReplied.addEventListener("change", () => {
  setReplied(elements.companyReplied.checked);
});
elements.deleteContact.addEventListener("click", deleteCurrentContact);
elements.saveResponse.addEventListener("click", saveCurrentResponse);
elements.fetchResponse.addEventListener("click", fetchCurrentResponse);
elements.loginOpen.addEventListener("click", showLogin);
elements.loginSkip.addEventListener("click", () => {
  elements.loginOverlay.hidden = true;
});
elements.loginForm.addEventListener("submit", submitLogin);
elements.logout.addEventListener("click", logout);

init();
