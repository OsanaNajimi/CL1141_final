// --------------------
// Reveal on scroll
// --------------------

const revealObserver = new IntersectionObserver(
  entries => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add("is-visible");
        revealObserver.unobserve(entry.target);
      }
    });
  },
  {
    threshold: 0.2
  }
);

document.querySelectorAll(".reveal").forEach((el, index) => {
  // 補一點自然延遲（卡片 / timeline / image 看起來更順）
  if (
    el.classList.contains("card") ||
    el.closest(".timeline") ||
    el.classList.contains("section__image")
  ) {
    el.style.transitionDelay = `${60 * (index % 4)}ms`;
  }
  revealObserver.observe(el);
});

// --------------------
// Hero parallax
// --------------------

const heroBg = document.querySelector(".hero__bg");
const heroContent = document.querySelector(".hero__content");

window.addEventListener("scroll", () => {
  const y = window.scrollY || window.pageYOffset;
  const limit = window.innerHeight * 1.2;

  if (heroBg && y < limit) {
    heroBg.style.transform = `translateY(${y * 0.15}px) scale(1.02)`;
  }
  if (heroContent && y < limit) {
    heroContent.style.transform = `translateY(${y * 0.06}px)`;
  }
});

// --------------------
// Smooth scroll for nav
// --------------------

document.querySelectorAll('.nav__links a[href^="#"]').forEach(link => {
  link.addEventListener("click", e => {
    e.preventDefault();
    const targetId = link.getAttribute("href").slice(1);
    const target = document.getElementById(targetId);
    if (!target) return;

    const navHeight = document.querySelector(".nav")?.offsetHeight || 0;
    const top =
      target.getBoundingClientRect().top + window.scrollY - navHeight - 24;

    window.scrollTo({ top, behavior: "smooth" });
  });
});

// --------------------
// Scroll Story：右邊 step 觸發左邊文字更新
// --------------------

const storyTitle = document.getElementById("story-main");
const storySteps = document.querySelectorAll(".story__step");

if (storyTitle && storySteps.length) {
  const storyObserver = new IntersectionObserver(
    entries => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          storySteps.forEach(step => step.classList.remove("is-active"));
          const current = entry.target;
          current.classList.add("is-active");

          const newText = current.dataset.text;
          if (newText && storyTitle.textContent !== newText) {
            storyTitle.style.opacity = 0;
            storyTitle.style.transform = "translateY(6px)";
            setTimeout(() => {
              storyTitle.textContent = newText;
              storyTitle.style.opacity = 1;
              storyTitle.style.transform = "translateY(0)";
            }, 160);
          }
        }
      });
    },
    {
      threshold: 0.55
    }
  );

  storySteps.forEach(step => storyObserver.observe(step));
}

// --------------------
// 表單提交：預留將來接語料庫 / AI API
// --------------------

const form = document.querySelector(".form-shell form");

if (form) {
  form.addEventListener("submit", async e => {
    e.preventDefault();

    const formData = new FormData(form);
    const occupation = formData.get("occupation")?.toString().trim() || "";
    const foreign = formData.get("foreign"); // yes / no
    const originalName =
      formData.get("originalName")?.toString().trim() || "";
    const meaning = formData.get("meaning")?.toString().trim() || "";
    const strokes = formData.get("strokes"); // yes / no

    const payload = {
      occupation,
      isForeign: foreign === "yes",
      originalName,
      meaningWish: meaning,
      useStrokeFortune: strokes === "yes"
    };

    const submitBtn = form.querySelector("button[type='submit']");
    const originalText = submitBtn.textContent;
    submitBtn.disabled = true;
    submitBtn.textContent = "生成中…";

    try {
      // 將來真的接語料庫 / 模型時，改成：
      // const res = await fetch("/api/generate-name", {
      //   method: "POST",
      //   headers: { "Content-Type": "application/json" },
      //   body: JSON.stringify(payload)
      // });
      // const data = await res.json();

      // 目前先 mock 一組假資料
      await new Promise(r => setTimeout(r, 800));

      const baseName = originalName || "Raymond Lee";
      const foreignLabel = payload.isForeign ? "外國人" : "本地使用";
      const strokeLabel = payload.useStrokeFortune ? "有考慮姓名結構" : "不拘泥筆畫";

      const data = {
        suggestions: [
          {
            name: "李睿安",
            type: "formal",
            reason: `適合作為正式中文名，強調穩定與智慧（${foreignLabel}、${strokeLabel}）。`
          },
          {
            name: "晉曜",
            type: "creative",
            reason: "光與前進的意象強烈，適合創作、社群或品牌使用。"
          },
          {
            name: "安行",
            type: "soft",
            reason: "語感溫柔，寓意「安然行走於世界」，適合低調但堅定的氣質。"
          }
        ],
        source: baseName
      };

      showNameResults(data, payload);
    } catch (err) {
      console.error(err);
      alert("生成時發生錯誤，之後可接上真正的 AI / 語料庫服務。");
    } finally {
      submitBtn.disabled = false;
      submitBtn.textContent = originalText;
    }
  });
}

function showNameResults(data, payload) {
  if (!data || !Array.isArray(data.suggestions)) return;

  let container = document.querySelector(".name-results");
  if (!container) {
    container = document.createElement("div");
    container.className = "name-results reveal";
    const formShell = document.querySelector(".form-shell");
    formShell?.appendChild(container);
    revealObserver.observe(container); // 也給它 reveal 動畫
  }

  const metaLineParts = [];
  if (payload.occupation) metaLineParts.push(`職業：${payload.occupation}`);
  if (payload.meaningWish)
    metaLineParts.push(`寓意：${payload.meaningWish}`);

  const metaLine =
    metaLineParts.length > 0 ? metaLineParts.join(" ｜ ") : "";

  container.innerHTML = `
    <h4>為你生成的名字：</h4>
    ${metaLine ? `<p style="font-size:12px;color:#888;margin:0 0 6px;">${metaLine}</p>` : ""}
    <ul>
      ${data.suggestions
      .map(
        s => `
        <li>
          <div class="name-results__name">${s.name}</div>
          <div class="name-results__meta">${s.reason || ""}</div>
        </li>
      `
      )
      .join("")}
    </ul>
  `;
}