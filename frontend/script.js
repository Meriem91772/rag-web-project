const API_URL = window.location.hostname
  ? window.location.origin
  : "http://127.0.0.1:8000";

async function askQuestion() {
  const questionInput = document.getElementById("question");
  const answerElem = document.getElementById("answer");
  const contextElem = document.getElementById("context");
  const scoreElem = document.getElementById("score");
  const question = questionInput.value.trim();

  if (!question) {
    answerElem.innerText = "Veuillez saisir une question.";
    return;
  }

  answerElem.innerText = "Chargement...";
  contextElem.innerText = "";
  scoreElem.innerText = "";

  try {
    const response = await fetch(`${API_URL}/ask`, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({question}),
    });

    if (!response.ok) {
      throw new Error(`Erreur API : ${response.status}`);
    }

    const data = await response.json();
    answerElem.innerText = data.answer || "Pas de réponse.";
    contextElem.innerText = data.context || "Pas de contexte.";
    scoreElem.innerText =
      data.score !== undefined ? Number(data.score).toFixed(3) : "Pas de score.";
  } catch (error) {
    console.error(error);
    answerElem.innerText =
      "Erreur : impossible de contacter l'API. Vérifiez que le backend est en ligne.";
  }
}
