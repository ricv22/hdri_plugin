const yearEl = document.getElementById("year");
if (yearEl) yearEl.textContent = String(new Date().getFullYear());

const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const isLowPowerDevice =
  (typeof navigator.hardwareConcurrency === "number" && navigator.hardwareConcurrency <= 4) ||
  (typeof navigator.deviceMemory === "number" && navigator.deviceMemory <= 4);
const useLiteMode = prefersReducedMotion || isLowPowerDevice;
if (useLiteMode) document.body.classList.add("lite-mode");

// ---------- i18n (Czech default) ----------
const translations = {
  cs: {
    "status": "Volný pro projekty",
    "nav.work": "Práce",
    "nav.art": "Art",
    "nav.about": "O mně",
    "nav.contact": "Kontakt",
    "hero.kicker": "3D artist — animace · matchmove · VFX",
    "hero.title": 'Tvořím <span class="fill">3D vizuály</span><br> pro obraz, značky a <span class="outline">příběhy</span>.',
    "hero.copy": "Animace, matchmove, produktové vizualizace a VFX. Od nápadu po finální záběr.",
    "cta.work": "Ukázat práce",
    "cta.start": "Napsat mi",
    "hero.reel": "pohybující se snímky :)",
    "cta.view": "Zobrazit projekt →",
    "label.work": "Vybrané projekty",
    "label.art": "Art výběr",
    "label.about": "O mně",
    "label.contact": "Kontakt",
    "usecases.line": "ANIMACE — EXPLAINERY — VIDEO KLIPY — VFX — KOMPOZIT — CAMERA TRACK — MATCHMOVE — MOTION DESIGN — PRODUKTOVÁ VIZ — CGI — REKLAMNÍ SPOTY — SOCIAL CONTENT — TITULKOVÉ SEKVENCE — LOOKDEV — SVÍCENÍ — SIMULACE — GRADING — RENDER — BRAND FILMY —",
    "tag.animation": "Animace",
    "tag.camera": "Kamera",
    "tag.realtime": "Realtime",
    "tag.productviz": "Produktová viz",
    "tag.motion": "Motion",
    "tag.lookdev": "Lookdev",
    "tag.lighting": "Svícení",
    "tag.vfx": "VFX",
    "tag.matchmove": "Matchmove",
    "tag.comp": "Kompozit",
    "tag.theatre": "Divadlo",
    "tag.ships": "Co záběr potřebuje",
    "proj.hangar": "FPV průlet lezeckou stěnou. Rychlost, pohyb, chaos pod kontrolou.",
    "proj.edisplay": "Produktové vizuály pro modulární výstavní stand.",
    "proj.haya": "Produktová vizualizace nápoje — jemné vlny, hluboká modrá, lesklá plechovka.",
    "proj.princess": "Filmové VFX — levitující dýka, tracking a kompozit.",
    "proj.rusalka": "Divadelní vizuály pro inscenaci Rusalka — atmosféra, světlo a pohyb na jevišti.",
    "detail.hangar.copy": "FPV průlet a kamerová choreografie pro sportovní prostor s důrazem na tempo a orientaci v prostoru.",
    "detail.edisplay.copy": "Produktová vizualizace modulárního standu. Klíčové bylo čisté nasvícení, čitelnost konstrukce a výrazná branding atmosféra.",
    "detail.haya.copy": "Produktová série nápoje Haya. Kombinace měkkého světla, materiality plechovky a barevné nálady pro premium feel.",
    "detail.princess.copy": "Cinematic VFX sekvence: tracking, digitální objekt, kompozit a grading pro konzistentní filmový look.",
    "detail.rusalka.copy": "Vizuální práce pro divadelní produkci Rusalka. Projekce, animace a světelná atmosféra pro podporu stage designu a dramaturgie.",
    "about.statement": 'Jsem <mark>3D artist</mark>. Tvořím animace, produktové vizuály a VFX — aby produkty vypadaly co nejlépe, <mark>snímek po snímku</mark>.',
    "contact.kicker": "Máš v hlavě záběr? Napiš.",
    "footer.built": "Richard Andrys — 3D artist",
    "modal.close": "Zavřít",
    "gate.title": "Kdo jste?",
    "gate.brand.label": "Jsme značka / firma",
    "gate.brand.copy": "Hledáme vizuály, díky kterým budou naše produkty atraktivní a nepřehlédnutelné.",
    "gate.agency.label": "Jsme agentura / studio",
    "gate.agency.copy": "Hledáme 3D parťáka do kampaní, postprodukce a náročnějších shotů.",
    "audience.brand": "Pro značky",
    "audience.agency": "Pro agentury",
    "audience.brand.short": "Značky",
    "audience.agency.short": "Agentury",
  },
  en: {
    "status": "Available for projects",
    "nav.work": "Work",
    "nav.art": "Art",
    "nav.about": "About",
    "nav.contact": "Contact",
    "hero.kicker": "3D artist — animation · matchmove · VFX",
    "hero.title": 'I create <span class="fill">3D visuals</span><br> for moving images, brands and <span class="outline">stories</span>.',
    "hero.copy": "Animation, matchmove, product visuals and VFX. From idea to final shot.",
    "cta.work": "See the work",
    "cta.start": "Start a project",
    "hero.reel": "moving frames :)",
    "cta.view": "View project →",
    "label.work": "Selected projects",
    "label.art": "Art selection",
    "label.about": "About",
    "label.contact": "Contact",
    "usecases.line": "ANIMATIONS — EXPLAINERS — VIDEO CLIPS — VFX — COMPOSITING — CAMERA TRACK — MATCHMOVE — MOTION DESIGN — PRODUCT VIZ — CGI — COMMERCIALS — SOCIAL CONTENT — TITLE SEQUENCES — LOOKDEV — LIGHTING — SIMULATION — GRADING — RENDER — BRAND FILMS —",
    "tag.animation": "Animation",
    "tag.camera": "Camera",
    "tag.realtime": "Realtime",
    "tag.productviz": "Product viz",
    "tag.motion": "Motion",
    "tag.lookdev": "Lookdev",
    "tag.lighting": "Lighting",
    "tag.vfx": "VFX",
    "tag.matchmove": "Matchmove",
    "tag.comp": "Comp",
    "tag.theatre": "Theatre",
    "tag.ships": "Whatever the shot needs",
    "proj.hangar": "FPV fly-through inside a climbing gym. Speed, movement, controlled chaos.",
    "proj.edisplay": "Product visuals for a modular exhibition stand.",
    "proj.haya": "Product visualization for a drink — soft waves, deep blue, glossy can.",
    "proj.princess": "Cinematic VFX — levitating dagger, tracking and compositing.",
    "proj.rusalka": "Theatre visuals for Rusalka — atmosphere, light and motion on stage.",
    "detail.hangar.copy": "FPV fly-through and camera choreography for a sport environment, focused on pace and spatial clarity.",
    "detail.edisplay.copy": "Product visualization for a modular stand. Priority was clean lighting, structure readability and strong branded atmosphere.",
    "detail.haya.copy": "Product series for Haya drink. Blend of soft lighting, can material detail and color mood for a premium feel.",
    "detail.princess.copy": "Cinematic VFX sequence: tracking, digital object integration, compositing and grading for a cohesive film look.",
    "detail.rusalka.copy": "Visual work for the Rusalka theatre production. Projection, animation and lighting atmosphere supporting stage design and dramaturgy.",
    "about.statement": 'I\'m a <mark>3D artist</mark>. I create animation, product visuals and VFX — making products look their best, <mark>one frame at a time</mark>.',
    "contact.kicker": "Have a shot in mind? Write me.",
    "footer.built": "Richard Andrys — 3D artist",
    "modal.close": "Close",
    "gate.title": "Who are you?",
    "gate.brand.label": "We're a brand / company",
    "gate.brand.copy": "We need visuals that make our products attractive and impossible to miss.",
    "gate.agency.label": "We're an agency / studio",
    "gate.agency.copy": "We need a 3D partner for campaigns, post and demanding shots.",
    "audience.brand": "For brands",
    "audience.agency": "For agencies",
    "audience.brand.short": "Brands",
    "audience.agency.short": "Agencies",
  },
};

let currentLang = "cs";
let activeProjectId = null;
let currentAudience = "agency";

const personaCopy = {
  brand: {
    cs: {
      "hero.kicker": "3D artist — fotorealistické produktové vizuály",
      "hero.title": 'Vizuály, které dávají <span class="fill">produktům</span> <span class="outline">charakter</span>.',
      "hero.copy": "Tvořím čisté a zapamatovatelné vizuální výstupy pro značky, kampaně i produkty – od nápadu až po finální záběr.",
      "usecases.line": "PRODUKTOVÉ VIZUÁLY — CGI KAMPANĚ — PACKSHOT ANIMACE — BRAND VIZUÁLY — PRODUKTOVÉ SPOTY — SOCIAL AD VIZUÁLY — KEY VISUALY — VIZUÁLY PRO LAUNCH PRODUKTŮ — 3D PRO E-SHOPY — MOTION DESIGN — VFX PRO REKLAMU — RETAIL KAMPANĚ — EXPLAINER ANIMACE — PRODUKTOVÉ RENDERY — CGI OBSAH — DIGITÁLNÍ PRODUKCE —",
    },
    en: {
      "hero.kicker": "3D artist — photoreal product visuals",
      "hero.title": 'Transform <span class="fill">products</span> into memorable <span class="outline">visual experiences</span>.',
      "hero.copy": "I help brands, campaigns and products look clean, memorable and impossible to ignore — from concept to final shot.",
      "usecases.line": "PRODUCT VISUALS — PACKSHOT ANIMATION — CGI CAMPAIGNS — PRODUCT RENDERS — BRAND LAUNCH VISUALS — SOCIAL AD CREATIVES — E-COMMERCE VISUALS — MOTION DESIGN — KEY VISUALS — EXPLAINER ANIMATION — RETAIL VISUALS — VFX FOR COMMERCIALS — CGI PRODUCTS — DIGITAL CAMPAIGN ASSETS —",
    },
  },
  agency: {
    cs: {
      "hero.kicker": "3D artist — animace · matchmove · VFX",
      "hero.title": 'Tvořím <span class="fill">3D vizuály</span> pro značky, kampaně a audiovizuální projekty.',
      "hero.copy": "Animace, matchmove, compositing, produktové vizualizace a VFX – pomáhám proměnit nápady ve finální záběry.",
      "usecases.line": "MATCHMOVE — CAMERA TRACKING — OBJECT TRACKING — ROTOSCOPING — KOMPOZIT — CLEANUP & PAINT — VFX INTEGRACE — CG INTEGRACE — MOTION TRACKING — KEYING — COLOR GRADING — STŘIH — ONLINE FINISHING — DOKONČENÍ SHOTŮ — PŘÍPRAVA PLATE — POSTPRODUKČNÍ PODPORA —",
    },
    en: {
      "hero.kicker": "3D artist — animation · matchmove · VFX",
      "hero.title": 'I create <span class="fill">3D visuals</span> for videos, brands and stories.',
      "hero.copy": "Animation, matchmove, compositing, product visuals and VFX — helping take ideas from concept to final shot.",
      "usecases.line": "MATCHMOVE — CAMERA TRACKING — OBJECT TRACKING — ROTOSCOPING — COMPOSITING — CLEANUP & PAINT — VFX INTEGRATION — CG INTEGRATION — MOTION TRACKING — KEYING — COLOR GRADING — EDITING — ONLINE FINISHING — SHOT FINISHING — PLATE PREPARATION — POST-PRODUCTION SUPPORT —",
    },
  },
};

const personaTargets = {
  "hero.kicker": document.querySelector('[data-persona-field="hero.kicker"]'),
  "hero.title": document.querySelector('[data-persona-field="hero.title"]'),
  "hero.copy": document.querySelector('[data-persona-field="hero.copy"]'),
  "usecases.line": document.querySelectorAll('[data-persona-field="usecases.line"]'),
};
const audienceGate = document.getElementById("audience-gate");
const audienceGateChoices = document.querySelectorAll(".audience-choice[data-audience]");
const audienceToggleButtons = document.querySelectorAll("[data-audience-toggle]");
const topbar = document.querySelector(".topbar");
const navToggle = document.getElementById("nav-toggle");
const navMenuLinks = document.querySelectorAll("#nav-menu a");

function setGateOpen(isOpen) {
  if (!audienceGate) return;
  audienceGate.classList.toggle("hidden", !isOpen);
  audienceGate.setAttribute("aria-hidden", isOpen ? "false" : "true");
  document.body.classList.toggle("gate-open", isOpen);
}

function applyAudienceCopy() {
  const audienceDict = (personaCopy[currentAudience] && personaCopy[currentAudience][currentLang]) || {};
  const defaultDict = translations[currentLang] || translations.cs;
  Object.entries(personaTargets).forEach(([key, target]) => {
    const value = audienceDict[key] || defaultDict[key];
    if (!value || !target) return;
    if (target instanceof NodeList) {
      target.forEach((node) => {
        node.innerHTML = value;
      });
      return;
    }
    target.innerHTML = value;
  });
}

function setAudience(audience, persist = true) {
  currentAudience = audience === "brand" ? "brand" : "agency";
  document.body.dataset.audience = currentAudience;
  audienceToggleButtons.forEach((btn) => {
    btn.classList.toggle("active", btn.getAttribute("data-audience-toggle") === currentAudience);
  });
  if (persist) {
    try { localStorage.setItem("audience", currentAudience); } catch (e) {}
  }
  applyAudienceCopy();
}

function setLang(lang) {
  const dict = translations[lang] || translations.cs;
  currentLang = lang;
  document.documentElement.lang = lang;
  document.querySelectorAll("[data-i18n]").forEach((el) => {
    const key = el.getAttribute("data-i18n");
    if (dict[key] != null) el.innerHTML = dict[key];
  });
  document.querySelectorAll("[data-lang]").forEach((btn) => {
    btn.classList.toggle("active", btn.getAttribute("data-lang") === lang);
  });
  applyAudienceCopy();
  if (activeProjectId) renderProjectModal(activeProjectId);
  try { localStorage.setItem("lang", lang); } catch (e) {}
}

const savedLang = (() => {
  try { return localStorage.getItem("lang"); } catch (e) { return null; }
})();

function detectBrowserLang() {
  const langs = navigator.languages?.length
    ? navigator.languages
    : [navigator.language || "cs"];
  for (const lang of langs) {
    const code = String(lang).toLowerCase().split("-")[0];
    if (code === "cs" || code === "en") return code;
  }
  return "cs";
}

const initialLang = savedLang === "en" || savedLang === "cs" ? savedLang : detectBrowserLang();
const savedAudience = (() => {
  try { return localStorage.getItem("audience"); } catch (e) { return null; }
})();
const urlAudience = new URLSearchParams(window.location.search).get("audience");
const initialAudience = urlAudience === "brand" || urlAudience === "agency"
  ? urlAudience
  : (savedAudience === "brand" || savedAudience === "agency" ? savedAudience : "agency");

setLang(initialLang);
setAudience(initialAudience, false);

document.querySelectorAll("[data-lang]").forEach((btn) => {
  btn.addEventListener("click", () => setLang(btn.getAttribute("data-lang")));
});
audienceGateChoices.forEach((btn) => {
  btn.addEventListener("click", (e) => {
    e.preventDefault();
    e.stopPropagation();
    setAudience(btn.getAttribute("data-audience"));
    setGateOpen(false);
  });
});
audienceToggleButtons.forEach((btn) => {
  btn.addEventListener("click", () => setAudience(btn.getAttribute("data-audience-toggle")));
});
if (audienceGate) {
  if (urlAudience === "brand" || urlAudience === "agency" || savedAudience === "brand" || savedAudience === "agency") {
    setGateOpen(false);
  } else {
    setGateOpen(true);
  }
}

if (topbar && navToggle) {
  navToggle.addEventListener("click", () => {
    const isOpen = topbar.classList.toggle("nav-open");
    navToggle.setAttribute("aria-expanded", isOpen ? "true" : "false");
  });
  navMenuLinks.forEach((link) => {
    link.addEventListener("click", () => {
      topbar.classList.remove("nav-open");
      navToggle.setAttribute("aria-expanded", "false");
    });
  });
  window.addEventListener("resize", () => {
    if (window.innerWidth > 420 && topbar.classList.contains("nav-open")) {
      topbar.classList.remove("nav-open");
      navToggle.setAttribute("aria-expanded", "false");
    }
  });
}

// ---------- custom cursor ----------
const cursor = document.querySelector(".cursor");
const fine = window.matchMedia("(hover: hover) and (pointer: fine)").matches;

if (cursor && fine && !useLiteMode) {
  let x = window.innerWidth / 2;
  let y = window.innerHeight / 2;
  let rafId = null;

  const renderCursor = () => {
    cursor.style.transform = `translate3d(${x}px, ${y}px, 0) translate(-50%, -50%)`;
    rafId = null;
  };

  const onPointerMove = (e) => {
    x = e.clientX;
    y = e.clientY;
    if (rafId == null) rafId = window.requestAnimationFrame(renderCursor);
  };

  window.addEventListener("pointermove", onPointerMove, { passive: true });
  renderCursor();

  document.querySelectorAll("[data-cursor]").forEach((el) => {
    const mode = el.getAttribute("data-cursor");
    el.addEventListener("mouseenter", () => {
      cursor.classList.add("grow");
      if (mode === "view") cursor.classList.add("view");
      if (mode === "play") cursor.classList.add("play");
    });
    el.addEventListener("mouseleave", () => {
      cursor.classList.remove("grow", "view", "play");
    });
  });
}

// ---------- project detail modal ----------
const projectLinks = document.querySelectorAll(".project[data-project]");
const projectModal = document.getElementById("project-modal");
const projectModalTitle = document.getElementById("project-modal-title");
const projectModalCopy = document.getElementById("project-modal-copy");
const projectModalGrid = document.getElementById("project-modal-grid");
const projectModalCloseEls = document.querySelectorAll("[data-modal-close]");

const projectDetails = {
  hangar: {
    title: "Hangar",
    copyKey: "detail.hangar.copy",
    images: ["hangar_01.png", "hangar_02.png", "hangar_03.png", "hangar_04.png"],
  },
  edisplay: {
    title: "eDisplay X-Stand",
    copyKey: "detail.edisplay.copy",
    images: ["edisplay_xstand_01.png", "edisplay_xstand_02.png", "edisplay_xstand_03.png"],
  },
  haya: {
    title: "Haya",
    copyKey: "detail.haya.copy",
    images: ["haya_01.png", "haya_02.png", "haya_03.png"],
  },
  princess: {
    title: "Princess Lost in Time",
    copyKey: "detail.princess.copy",
    images: ["princes_01.png", "princes_02.png", "princes_03.png"],
  },
  rusalka: {
    title: "Rusalka",
    copyKey: "detail.rusalka.copy",
    videos: ["rusalka_reel.mp4"],
    images: ["rusalka_1.jpg", "rusalka_2.jpg", "rusalka_3.jpg"],
  },
};

function renderProjectModal(projectId) {
  if (!projectModal || !projectModalTitle || !projectModalCopy || !projectModalGrid) return;
  const project = projectDetails[projectId];
  if (!project) return;
  const dict = translations[currentLang] || translations.cs;
  projectModalTitle.textContent = project.title;
  projectModalCopy.innerHTML = dict[project.copyKey] || "";
  const videoHtml = (project.videos || [])
    .map((vid) => `<video class="project-modal__video" src="./assets/${vid}" autoplay muted loop playsinline preload="auto"></video>`)
    .join("");
  const imageHtml = (project.images || [])
    .map((img, i) => `<img src="./assets/${img}" alt="${project.title} detail ${i + 1}" loading="lazy">`)
    .join("");
  projectModalGrid.innerHTML = imageHtml + videoHtml;
}

function openProjectModal(projectId) {
  if (!projectModal) return;
  activeProjectId = projectId;
  renderProjectModal(projectId);
  projectModal.classList.add("open");
  projectModal.setAttribute("aria-hidden", "false");
  document.body.style.overflow = "hidden";
  projectModal.querySelectorAll("video").forEach((v) => v.play().catch(() => {}));
}

function closeProjectModal() {
  if (!projectModal) return;
  activeProjectId = null;
  projectModal.classList.remove("open");
  projectModal.setAttribute("aria-hidden", "true");
  document.body.style.overflow = "";
}

projectLinks.forEach((link) => {
  link.addEventListener("click", (e) => {
    e.preventDefault();
    const projectId = link.getAttribute("data-project");
    if (!projectId) return;
    openProjectModal(projectId);
  });
});

projectModalCloseEls.forEach((el) => {
  el.addEventListener("click", closeProjectModal);
});

window.addEventListener("keydown", (e) => {
  if (e.key === "Escape") closeProjectModal();
});

// ---------- scroll reveal ----------
const revealEls = document.querySelectorAll(".reveal");
const io = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add("in");
        io.unobserve(entry.target);
      }
    });
  },
  { threshold: 0.15 }
);
revealEls.forEach((el) => io.observe(el));

// ---------- keep reel(s) playing ----------
const reels = document.querySelectorAll("video");
const tryPlay = () => reels.forEach((v) => v.play().catch(() => {}));
tryPlay();
document.addEventListener("visibilitychange", () => {
  if (!document.hidden) tryPlay();
});
window.addEventListener("click", tryPlay, { once: true });
