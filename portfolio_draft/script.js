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
    "about.statement": 'Jsem <mark>3D artist</mark>. Dělám animace, matchmove, produktové vizualizace a VFX — od čistého renderu po <mark>finální kompozit</mark>.',
    "contact.kicker": "Máš v hlavě záběr? Napiš.",
    "footer.built": "Richard Andrys — 3D artist",
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
    "about.statement": "I'm a <mark>3D artist</mark>. I create animation, matchmove, product visuals and VFX — from clean render to <mark>final composite</mark>.",
    "contact.kicker": "Have a shot in mind? Write me.",
    "footer.built": "Richard Andrys — 3D artist",
  },
};

function setLang(lang) {
  const dict = translations[lang] || translations.cs;
  document.documentElement.lang = lang;
  document.querySelectorAll("[data-i18n]").forEach((el) => {
    const key = el.getAttribute("data-i18n");
    if (dict[key] != null) el.innerHTML = dict[key];
  });
  document.querySelectorAll("[data-lang]").forEach((btn) => {
    btn.classList.toggle("active", btn.getAttribute("data-lang") === lang);
  });
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
  let cx = x;
  let cy = y;

  window.addEventListener("mousemove", (e) => {
    x = e.clientX;
    y = e.clientY;
  });

  const render = () => {
    cx += (x - cx) * 0.2;
    cy += (y - cy) * 0.2;
    cursor.style.transform = `translate(${cx}px, ${cy}px) translate(-50%, -50%)`;
    requestAnimationFrame(render);
  };
  render();

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
