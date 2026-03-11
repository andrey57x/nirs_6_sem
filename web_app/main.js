import {
  AutoTokenizer,
  CLIPTextModelWithProjection,
  CLIPVisionModelWithProjection,
  AutoProcessor,
  RawImage,
  env,
} from "https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.1";

env.allowLocalModels = false;

let tokenizer, textModel, visionModel, processor;
let database = [];

const statusEl = document.getElementById("status");
const searchInput = document.getElementById("searchInput");
const searchBtn = document.getElementById("searchBtn");
const resultsGrid = document.getElementById("resultsGrid");

const compareTextInput = document.getElementById("compareTextInput");
const compareBtn = document.getElementById("compareBtn");
const compareResultValue = document.getElementById("compareResultValue");

// 1. Переключение режимов
document.getElementById("btnModeSearch").onclick = () => {
    document.getElementById("sectionSearch").style.display = "block";
    document.getElementById("sectionCompare").style.display = "none";
    document.getElementById("btnModeSearch").classList.add("active");
    document.getElementById("btnModeCompare").classList.remove("active");
};
document.getElementById("btnModeCompare").onclick = () => {
    document.getElementById("sectionSearch").style.display = "none";
    document.getElementById("sectionCompare").style.display = "block";
    document.getElementById("btnModeSearch").classList.remove("active");
    document.getElementById("btnModeCompare").classList.add("active");
};

// 2. Классическая формула косинусного сходства
function rawCosineSimilarity(vecA, vecB) {
    let dotProduct = 0;
    let normA = 0;
    let normB = 0;
    for (let i = 0; i < vecA.length; i++) {
        dotProduct += vecA[i] * vecB[i];
        normA += vecA[i] * vecA[i];
        normB += vecB[i] * vecB[i];
    }
    const similarity = dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
    return similarity;
}

// 3. Загрузка
async function init() {
  try {
    statusEl.innerText = "Загрузка базы векторов...";
    const response = await fetch("public/data.json");
    database = await response.json();

    statusEl.innerText = "Загрузка моделей CLIP...";
    tokenizer = await AutoTokenizer.from_pretrained("Xenova/clip-vit-base-patch32");
    textModel = await CLIPTextModelWithProjection.from_pretrained("Xenova/clip-vit-base-patch32", { model_file: "text_model" });
    
    visionModel = await CLIPVisionModelWithProjection.from_pretrained("Xenova/clip-vit-base-patch32", { model_file: "vision_model" });
    processor = await AutoProcessor.from_pretrained("Xenova/clip-vit-base-patch32");

    statusEl.innerText = "Система готова к работе!";
    searchInput.disabled = false;
    searchBtn.disabled = false;
  } catch (e) {
    statusEl.innerText = "Ошибка загрузки: " + e.message;
  }
}

// 4. Поиск по базе
async function performSearch() {
  const query = searchInput.value;
  if (!query) return;

  statusEl.innerText = "Кодирование текста...";
  const textInputs = tokenizer(`A photo of ${query}`, { padding: true, truncation: true });
  const { text_embeds } = await textModel(textInputs);
  const queryVector = Array.from(text_embeds.data);

  statusEl.innerText = "Поиск...";
  const scoredResults = database.map((item) => ({
    ...item,
    score: rawCosineSimilarity(queryVector, item.vector),
  }));

  const topResults = scoredResults.sort((a, b) => b.score - a.score).slice(0, 16);
  renderResults(topResults);
  statusEl.innerText = `Найдено. Объектов в базе: ${database.length}`;
}

// 5. Режим сравнения (Baseline)
async function performCompare() {
    const file = document.getElementById("imageUpload").files[0];
    const text = compareTextInput.value;
    if (!file || !text) return alert("Нужна картинка и текст");

    statusEl.innerText = "Обработка...";
    
    // Вектор текста
    const textInputs = tokenizer(`A photo of ${text}`, { padding: true, truncation: true });
    const { text_embeds } = await textModel(textInputs);
    const textVec = Array.from(text_embeds.data);

    // Вектор картинки
    const image = await RawImage.fromURL(URL.createObjectURL(file));
    const visionInputs = await processor(image);
    const { image_embeds } = await visionModel(visionInputs);
    const imgVec = Array.from(image_embeds.data);

    const score = rawCosineSimilarity(textVec, imgVec);
    compareResultValue.innerText = `Сходство: ${(score * 100).toFixed(2)}%`;
    statusEl.innerText = "Готово";
}

function renderResults(items) {
  resultsGrid.innerHTML = items.map(item => `
    <div class="card">
        <img src="public/images/${item.image}">
        <div class="card-info">
            <h3>${item.name}</h3>
            <p>Score: ${(item.score * 100).toFixed(2)}%</p>
        </div>
    </div>
  `).join("");
}

searchBtn.onclick = performSearch;
compareBtn.onclick = performCompare;
searchInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !searchBtn.disabled) {
        performSearch();
    }
});
compareTextInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
        performCompare();
    }
});
init();
