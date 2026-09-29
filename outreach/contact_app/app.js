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
  companyCard: document.querySelector(".company-card"),
};

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
  return contact.outreach_language.toLowerCase().startsWith("czech") ? "cz" : "en";
}

function applyFilters() {
  const query = state.query.trim().toLowerCase();
  state.filtered = state.contacts.filter((contact) => {
    const rank = Number(contact.priority);
    const priorityMatch =
      state.priority === "all" ||
      (state.priority === "5" && rank === 5) ||
      (state.priority === "4" && rank >= 4);
    const reached = isReached(contact.company);
    const reachMatch =
      state.reachView === "all" ||
      (state.reachView === "reached" && reached) ||
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
  elements.companyCard.classList.toggle("is-reached", isReached(contact.company));
  if (state.renderedCompany !== contact.company) {
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
    : "Copy only until SMTP_USER and SMTP_PASSWORD are set";
  elements.meta.textContent = `${recipient} · ${subject} · ${sendHint}`;
  elements.sendMessage.disabled = !state.sendConfigured;
  elements.sendMessage.title = state.sendConfigured
    ? `Send this one email as ${state.sendFrom}`
    : "Set SMTP_USER and SMTP_PASSWORD before sending";
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
    elements.sendMessage.disabled = !state.sendConfigured;
  }
}

async function toggleReached(company, reached) {
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

async function init() {
  try {
    const [contactsResponse, statusResponse, reachedResponse] = await Promise.all([
      fetch("/api/contacts", { cache: "no-store" }),
      fetch("/api/send-status", { cache: "no-store" }),
      fetch("/api/reached", { cache: "no-store" }),
    ]);
    if (!contactsResponse.ok) throw new Error(`HTTP ${contactsResponse.status}`);
    const data = await contactsResponse.json();
    if (statusResponse.ok) {
      const status = await statusResponse.json();
      state.sendConfigured = Boolean(status.configured);
      state.sendFrom = status.from || "";
    }
    if (reachedResponse.ok) {
      applyReached(await reachedResponse.json());
    }
    state.contacts = data.contacts;
    state.filtered = data.contacts;
    state.selected = data.contacts[0] || null;
    if (state.selected) state.language = defaultLanguage(state.selected);
    applyFilters();
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

init();
