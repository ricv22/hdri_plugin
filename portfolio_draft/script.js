const yearEl = document.getElementById("year");
if (yearEl) yearEl.textContent = String(new Date().getFullYear());

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
    "tag.houdini": "Houdini (učím se)",
    "tag.ships": "Co záběr potřebuje",
    "proj.hangar": "FPV průlet lezeckou stěnou. Rychlost, pohyb, chaos pod kontrolou.",
    "proj.edisplay": "Produktové vizuály pro modulární výstavní stand.",
    "proj.haya": "Produktová vizualizace nápoje — jemné vlny, hluboká modrá, lesklá plechovka.",
    "proj.princess": "Filmové VFX — levitující dýka, tracking a kompozit.",
    "detail.hangar.copy": "FPV průlet a kamerová choreografie pro sportovní prostor s důrazem na tempo a orientaci v prostoru.",
    "detail.edisplay.copy": "Produktová vizualizace modulárního standu. Klíčové bylo čisté nasvícení, čitelnost konstrukce a výrazná branding atmosféra.",
    "detail.haya.copy": "Produktová série nápoje Haya. Kombinace měkkého světla, materiality plechovky a barevné nálady pro premium feel.",
    "detail.princess.copy": "Cinematic VFX sekvence: tracking, digitální objekt, kompozit a grading pro konzistentní filmový look.",
    "about.statement": 'Jsem <mark>3D artist</mark>. Dělám animace, matchmove, produktové vizualizace a VFX — od čistého renderu po <mark>finální kompozit</mark>.',
    "contact.kicker": "Máš v hlavě záběr? Napiš.",
    "footer.built": "Richard Andrys — 3D artist",
    "modal.close": "Zavřít",
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
    "tag.houdini": "Houdini-curious",
    "tag.ships": "Whatever the shot needs",
    "proj.hangar": "FPV fly-through inside a climbing gym. Speed, movement, controlled chaos.",
    "proj.edisplay": "Product visuals for a modular exhibition stand.",
    "proj.haya": "Product visualization for a drink — soft waves, deep blue, glossy can.",
    "proj.princess": "Cinematic VFX — levitating dagger, tracking and compositing.",
    "detail.hangar.copy": "FPV fly-through and camera choreography for a sport environment, focused on pace and spatial clarity.",
    "detail.edisplay.copy": "Product visualization for a modular stand. Priority was clean lighting, structure readability and strong branded atmosphere.",
    "detail.haya.copy": "Product series for Haya drink. Blend of soft lighting, can material detail and color mood for a premium feel.",
    "detail.princess.copy": "Cinematic VFX sequence: tracking, digital object integration, compositing and grading for a cohesive film look.",
    "about.statement": "I'm a <mark>3D artist</mark>. I create animation, matchmove, product visuals and VFX — from clean render to <mark>final composite</mark>.",
    "contact.kicker": "Have a shot in mind? Write me.",
    "footer.built": "Richard Andrys — 3D artist",
    "modal.close": "Close",
  },
};

let currentLang = "cs";
let activeProjectId = null;

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
  if (activeProjectId) renderProjectModal(activeProjectId);
  try { localStorage.setItem("lang", lang); } catch (e) {}
}

const savedLang = (() => {
  try { return localStorage.getItem("lang"); } catch (e) { return null; }
})();
setLang(savedLang === "en" ? "en" : "cs");

document.querySelectorAll("[data-lang]").forEach((btn) => {
  btn.addEventListener("click", () => setLang(btn.getAttribute("data-lang")));
});

// ---------- custom cursor ----------
const cursor = document.querySelector(".cursor");
const fine = window.matchMedia("(hover: hover) and (pointer: fine)").matches;

if (cursor && fine) {
  let x = window.innerWidth / 2;
  let y = window.innerHeight / 2;

  window.addEventListener("mousemove", (e) => {
    x = e.clientX;
    y = e.clientY;
    cursor.style.transform = `translate(${x}px, ${y}px) translate(-50%, -50%)`;
  });
  cursor.style.transform = `translate(${x}px, ${y}px) translate(-50%, -50%)`;

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
    images: ["hangar_01.png", "hangar_02.png", "hangar_registrace_01.png", "hangar_registrace_02.png"],
  },
  edisplay: {
    title: "eDisplay X-Stand",
    copyKey: "detail.edisplay.copy",
    images: ["edisplay_xstand.png"],
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
};

function renderProjectModal(projectId) {
  if (!projectModal || !projectModalTitle || !projectModalCopy || !projectModalGrid) return;
  const project = projectDetails[projectId];
  if (!project) return;
  const dict = translations[currentLang] || translations.cs;
  projectModalTitle.textContent = project.title;
  projectModalCopy.innerHTML = dict[project.copyKey] || "";
  projectModalGrid.innerHTML = project.images
    .map((img, i) => `<img src="./assets/${img}" alt="${project.title} detail ${i + 1}" loading="lazy">`)
    .join("");
}

function openProjectModal(projectId) {
  if (!projectModal) return;
  activeProjectId = projectId;
  renderProjectModal(projectId);
  projectModal.classList.add("open");
  projectModal.setAttribute("aria-hidden", "false");
  document.body.style.overflow = "hidden";
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
