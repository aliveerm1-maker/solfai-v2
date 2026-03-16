// server.js — Solfai Express server for Render.com
// Gemini 2.5 Pro with full thinking (Render free tier = 30s timeout)

import express from 'express';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const app = express();
const PORT = process.env.PORT || 3000;

// Parse JSON bodies up to 20MB (PDF pages as base64)
app.use(express.json({ limit: '20mb' }));

// Serve static files (index.html etc)
app.use(express.static(join(__dirname, 'public')));

// ─── Config ───────────────────────────────────────────────
const GEMINI_MODEL = 'gemini-2.5-pro';
const GEMINI_URL = `https://generativelanguage.googleapis.com/v1beta/models/${GEMINI_MODEL}:generateContent`;

// ─── API Route ────────────────────────────────────────────
app.post('/api/analyze', async (req, res) => {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    return res.status(500).json({ error: 'GEMINI_API_KEY not configured' });
  }

  try {
    const { messages, imageBase64, imageMime, pdfPages, mode, selectedPart } = req.body;
    const part = selectedPart || 'Soprano';

    const imageParts = buildImageParts(imageBase64, imageMime, pdfPages);
    if (!imageParts.length) {
      return res.status(400).json({ error: 'No image provided' });
    }

    const totalSize = imageParts.reduce((s, p) => s + (p.inlineData?.data?.length || 0), 0);
    console.log(`[Solfai] mode=${mode}, part=${part}, images=${imageParts.length}, size=${(totalSize/1024/1024).toFixed(1)}MB`);

    switch (mode) {
      case 'analyze':  return await handleAnalyze(res, apiKey, imageParts, part);
      case 'solfege':  return await handleSolfege(res, apiKey, imageParts, part);
      case 'rhythm':   return await handleRhythm(res, apiKey, imageParts, part);
      case 'chat':     return await handleChat(res, apiKey, messages, imageParts, part);
      default:         return res.status(400).json({ error: 'Invalid mode' });
    }
  } catch (err) {
    console.error('Handler error:', err.message);
    return res.status(500).json({ error: err.message || 'Internal server error' });
  }
});

// ─── Image part builder ───────────────────────────────────
function buildImageParts(imageBase64, imageMime, pdfPages) {
  const parts = [];
  if (pdfPages && pdfPages.length > 0) {
    for (const page of pdfPages) {
      parts.push({ inlineData: { mimeType: 'image/jpeg', data: page } });
    }
  } else if (imageBase64) {
    parts.push({ inlineData: { mimeType: imageMime || 'image/jpeg', data: imageBase64 } });
  }
  return parts;
}

// ─── Gemini caller ────────────────────────────────────────
async function callGemini(apiKey, systemPrompt, userParts, opts = {}) {
  const {
    temperature = 0.1,
    thinkingBudget = 8000,
    maxOutputTokens = 16384,
    model = GEMINI_MODEL,
  } = opts;

  const url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${apiKey}`;

  const body = {
    contents: [{ role: 'user', parts: userParts }],
    systemInstruction: { parts: [{ text: systemPrompt }] },
    generationConfig: {
      temperature,
      maxOutputTokens,
      thinkingConfig: { thinkingBudget },
    },
  };

  const resp = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });

  if (!resp.ok) {
    const errText = await resp.text();
    console.error(`Gemini API error: ${resp.status}`, errText.substring(0, 500));
    let detail = `Gemini API error: ${resp.status}`;
    try { detail = JSON.parse(errText).error?.message || detail; } catch (_) {}
    throw new Error(detail);
  }

  const data = await resp.json();
  const candidate = data.candidates?.[0];
  if (!candidate?.content?.parts) throw new Error('No response from Gemini');

  return candidate.content.parts
    .filter(p => p.text && !p.thought)
    .map(p => p.text)
    .join('')
    .trim();
}

// ─── ANALYZE ──────────────────────────────────────────────
async function handleAnalyze(res, apiKey, imageParts, part) {
  const systemPrompt = `You are a music theory expert reading sheet music images. Be precise. Only report what you see.
If any image is a title/cover page with no music notation, SKIP IT and analyze the image that contains actual sheet music.

## KEY SIGNATURE
Count flats/sharps between the clef and time signature, then use this EXACT mapping:
0 = C major/A minor | 1♭ = F major/D minor | 2♭ = B♭ major/G minor | 3♭ = E♭ major/C minor | 4♭ = A♭ major/F minor | 5♭ = D♭ major/B♭ minor | 1♯ = G major/E minor | 2♯ = D major/B minor | 3♯ = A major/F♯ minor | 4♯ = E major/C♯ minor | 5♯ = B major/G♯ minor
IMPORTANT: 1 flat = F major or D minor (NEVER C major). 0 flats = C major or A minor (NEVER F major).
MAJOR vs MINOR: Default to MAJOR unless strong evidence of minor (piece starts/ends on minor chord, raised 7th, dark character). Most choral and art song repertoire is in major keys.
If this is a well-known piece, use your knowledge: e.g. Fauré "Lydia" = F major, Schubert "Ave Maria" = B♭ major, Mozart "Laudate Dominum" = F major.

## STARTING PITCH
1. Find the VOCAL staff (has lyrics/words printed underneath). Skip any piano introduction — find where the first LYRIC SYLLABLE appears.
2. Note the syllable under the first sung note.
3. Read the note:
   TREBLE CLEF — Lines bottom→top: E4 G4 B4 D5 F5 | Spaces bottom→top: F4 A4 C5 E5
   Below staff: D4 (just below 1st line), C4/middle C (1st ledger line below)
   BASS CLEF — Lines: G2 B2 D3 F3 A3 | Spaces: A2 C3 E3 G3
   For SATB: Soprano=top treble stems up, Alto=bottom treble stems down, Tenor=top bass stems up, Bass=bottom bass stems down
4. Cross-check: if well-known piece, verify against your knowledge (e.g. Fauré "Lydia" starts on A4).

## LYRICS
Quote ONLY exact text from the score. Never invent words. Use [unclear] for illegible text.

## OUTPUT: valid JSON only, no markdown fencing.
{
  "keySignature": "e.g. F major (1 flat)",
  "timeSignature": "e.g. 4/4",
  "tempo": "e.g. Andante",
  "dynamics": "e.g. Starts p at m.3",
  "startingPitch": "e.g. A4 (2nd space treble clef, syllable 'Ly')",
  "difficulty": { "overall": 5, "rhythm": 4, "range": 3, "intervals": 4 },
  "overview": "2-3 paragraphs for a choir student about the piece",
  "practiceTips": ["tip 1", "tip 2", "tip 3", "tip 4", "tip 5"],
  "composerName": "name or null",
  "composerBio": "2 sentences or null",
  "pieceTitle": "title or null",
  "pieceInfo": "context or null",
  "pronunciation": {
    "language": "French/Latin/German/English/etc",
    "needsGuide": true,
    "words": [{ "word": "original", "ipa": "/ipa/", "approx": "sounds-LIKE" }]
  }
}

For pronunciation: include EVERY unique word from visible lyrics with IPA and English approximation. For straightforward English, set needsGuide to false and words to [].`;

  // Send up to 3 pages — Pro can handle it in 30s
  const limitedParts = imageParts.length > 3 ? imageParts.slice(0, 3) : imageParts;

  const userParts = [
    { text: `Analyze this sheet music for the ${part} part. Output ONLY valid JSON.` },
    ...limitedParts,
  ];

  const raw = await callGemini(apiKey, systemPrompt, userParts, {
    temperature: 0.05,
    thinkingBudget: 8000,
    maxOutputTokens: 12288,
  });

  let structured;
  try {
    const cleaned = raw.replace(/^```(?:json)?\s*/i, '').replace(/\s*```$/i, '').trim();
    structured = JSON.parse(cleaned);
  } catch (parseErr) {
    console.error('JSON parse failed:', parseErr.message, raw.substring(0, 300));
    return res.status(200).json({ text: raw });
  }

  structured = {
    keySignature: structured.keySignature || 'Not determined',
    timeSignature: structured.timeSignature || 'Not determined',
    tempo: structured.tempo || 'Not marked',
    dynamics: structured.dynamics || 'None visible',
    startingPitch: structured.startingPitch || 'Not determined',
    difficulty: {
      overall: Number(structured.difficulty?.overall) || 5,
      rhythm: Number(structured.difficulty?.rhythm) || 5,
      range: Number(structured.difficulty?.range) || 5,
      intervals: Number(structured.difficulty?.intervals) || 5,
    },
    overview: structured.overview || '',
    practiceTips: Array.isArray(structured.practiceTips) ? structured.practiceTips : [],
    composerName: structured.composerName || null,
    composerBio: structured.composerBio || null,
    pieceTitle: structured.pieceTitle || null,
    pieceInfo: structured.pieceInfo || null,
    pronunciation: structured.pronunciation || { language: 'English', needsGuide: false, words: [] },
  };

  const textSummary = buildTextSummary(structured, part);
  return res.status(200).json({ structured, text: textSummary });
}

function buildTextSummary(s, part) {
  return [
    `Key Signature: ${s.keySignature}`,
    `Time Signature: ${s.timeSignature}`,
    `Tempo: ${s.tempo}`,
    `Dynamics: ${s.dynamics}`,
    `Starting Pitch (${part}): ${s.startingPitch}`,
    `Difficulty Overall: ${s.difficulty.overall}/10`,
    `Difficulty Rhythm: ${s.difficulty.rhythm}/10`,
    `Difficulty Range: ${s.difficulty.range}/10`,
    `Difficulty Intervals: ${s.difficulty.intervals}/10`,
    `---`, `BREAKDOWN:`, s.overview,
    `---`, `PRACTICE TIPS:`,
    ...s.practiceTips.map((t, i) => `${i + 1}. ${t}`),
    s.composerName ? `COMPOSER: ${s.composerName}. ${s.composerBio || ''}` : '',
    s.pieceTitle ? `PIECE INFO: ${s.pieceTitle}. ${s.pieceInfo || ''}` : '',
  ].filter(Boolean).join('\n');
}

// ─── SOLFEGE ──────────────────────────────────────────────
async function handleSolfege(res, apiKey, imageParts, part) {
  const systemPrompt = `You are a sight-reading coach generating solfege for choir students.

RULES:
- Use MOVABLE DO. Determine key first, assign Do to tonic.
- Go measure by measure through ONLY measures visible in the images.
- For each measure, list ONLY notes of the ${part} voice.
- Include ACTUAL lyrics from the score. If unclear, write [unclear]. NEVER invent.
- If you can't read a note, write [?].
- First full measure = m.1 (pickup = m.0).
- If an image is a title page, skip it.

SATB: Soprano=top treble stems up, Alto=bottom treble stems down, Tenor=top bass stems up, Bass=bottom bass stems down.

OUTPUT:
Key: [key] (Do = [note])

m.1: [solfege with | for beats] | lyrics: "[text]"
m.2: ...

Syllables — Major: Do Re Mi Fa Sol La Ti | Chromatic: Di Ri Fi Si Li (up) Ra Me Se Le Te (down)`;

  const limitedParts = imageParts.length > 3 ? imageParts.slice(0, 3) : imageParts;
  const userParts = [
    { text: `Generate measure-by-measure solfege for the ${part} part. Only include what you can see.` },
    ...limitedParts,
  ];

  const text = await callGemini(apiKey, systemPrompt, userParts, {
    thinkingBudget: 6000,
  });
  return res.status(200).json({ text });
}

// ─── RHYTHM ───────────────────────────────────────────────
async function handleRhythm(res, apiKey, imageParts, part) {
  const systemPrompt = `You are a rhythm coach for choir students.

RULES:
- Analyze ONLY the ${part} voice. Reference ONLY visible measures.
- If an image is a title page, skip it.
- SATB: Soprano=top treble stems up, Alto=bottom treble stems down, Tenor=top bass stems up, Bass=bottom bass stems down.

OUTPUT:
State time signature and what gets one beat.
For each measure:
m.X: Count: "1 + 2 + 3 + 4 +" | Notes: [what ${part} sings] | Tips: [tricky spots]

Counting: 4/4="1 + 2 + 3 + 4 +" | 3/4="1 + 2 + 3 +" | 6/8="1-la-li 2-la-li"`;

  const limitedParts = imageParts.length > 3 ? imageParts.slice(0, 3) : imageParts;
  const userParts = [
    { text: `Create a rhythm guide for the ${part} part. Only analyze what you can see.` },
    ...limitedParts,
  ];

  const text = await callGemini(apiKey, systemPrompt, userParts, {
    thinkingBudget: 6000,
  });
  return res.status(200).json({ text });
}

// ─── CHAT ─────────────────────────────────────────────────
async function handleChat(res, apiKey, messages, imageParts, part) {
  const systemPrompt = `You are Solfai, a patient choir director / music theory coach. The student sings ${part}.
RULES: Reference actual sheet music in the images. Only cite visible measures/notes/lyrics. Use plain, encouraging language. Include solfege (movable Do) when relevant. Never invent content not in the images. If an image is a title page, ignore it.
SATB: Soprano=top treble stems up, Alto=bottom treble stems down, Tenor=top bass stems up, Bass=bottom bass stems down.`;

  const contents = [];
  const chatImages = imageParts.length > 2 ? imageParts.slice(0, 2) : imageParts;
  let attached = false;

  if (messages?.length > 0) {
    for (const msg of messages) {
      if (msg.role === 'user') {
        const parts = [{ text: msg.parts?.[0]?.text || msg.text || '' }];
        if (!attached) { parts.push(...chatImages); attached = true; }
        contents.push({ role: 'user', parts });
      } else if (msg.role === 'model') {
        contents.push({ role: 'model', parts: [{ text: msg.parts?.[0]?.text || msg.text || '' }] });
      }
    }
  }
  if (!contents.length) {
    contents.push({ role: 'user', parts: [{ text: 'Help me with this sheet music.' }, ...chatImages] });
  }

  const body = {
    contents,
    systemInstruction: { parts: [{ text: systemPrompt }] },
    generationConfig: {
      temperature: 0.3,
      maxOutputTokens: 4096,
      thinkingConfig: { thinkingBudget: 4000 },
    },
  };

  const resp = await fetch(`${GEMINI_URL}?key=${apiKey}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });

  if (!resp.ok) {
    const errText = await resp.text();
    console.error(`Gemini chat error: ${resp.status}`, errText.substring(0, 500));
    let detail = `Gemini error: ${resp.status}`;
    try { detail = JSON.parse(errText).error?.message || detail; } catch (_) {}
    throw new Error(detail);
  }

  const data = await resp.json();
  const text = data.candidates?.[0]?.content?.parts
    ?.filter(p => p.text && !p.thought)
    .map(p => p.text)
    .join('') || '';

  return res.status(200).json({ text: text.trim() });
}

// ─── Start ────────────────────────────────────────────────
app.listen(PORT, () => {
  console.log(`Solfai server running on port ${PORT}`);
});
