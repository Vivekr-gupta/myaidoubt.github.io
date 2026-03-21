// ============================================================
// MyAIDoubt — Smart Worker
// Routing: Maths/Physics/Chemistry → Gemini 2.5 Flash
//          Others → Groq llama-3.3-70b
// Photo: Scout (read) → smart route (solve)
// ============================================================

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type",
};

// ── Keyword Lists ──────────────────────────────────────────

const mathKeywords = [
  "integral","integration","differentiate","derivative","limit","continuity","function","domain","range",
  "matrix","determinant","vector","dot product","cross product","3d geometry","coordinate geometry",
  "circle","parabola","ellipse","hyperbola","straight line","slope","distance","section formula",
  "trigonometry","sin","cos","tan","cot","sec","cosec","identity","equation","solution",
  "quadratic","roots","polynomial","inequality","modulus","log","logarithm","exponential",
  "series","sequence","ap","gp","hp","binomial","probability","permutation","combination",
  "mean","median","variance","standard deviation","graph","area under curve","differential equation",
  "complex number","argand","real","imaginary","locus","inverse function","composition",
  // extra math operators/symbols context
  "solve","calculate","find the value","evaluate","simplify","expand","factorise","factorize",
  "prove","verify","theta","alpha","beta","pi","infinity","sigma","summation","arithmetic","geometric",
  // geometry & mensuration
  "area","perimeter","volume","surface area","radius","diameter","circumference","hypotenuse",
  "triangle","rectangle","square","cube","cone","cylinder","sphere","rhombus","trapezium",
  "angle","bisector","altitude","centroid","orthocenter","incenter",
  // number system & applied maths
  "lcm","hcf","gcd","prime","factor","multiple","fraction","decimal","percentage","ratio","proportion",
  "profit","loss","discount","interest","simple interest","compound interest",
  // symbols as words
  "plus","minus","divide","multiply","power","root","square root","cube root","bracket",
  "pi value","value of pi","3.14"
];

const physicsKeywords = [
  "velocity","acceleration","displacement","speed","force","newton","friction","tension",
  "work","energy","power","kinetic","potential","momentum","impulse","collision",
  "rotation","angular","torque","moment of inertia","rolling","circular motion",
  "gravitation","gravity","orbital","escape velocity","satellite",
  "electric field","electric potential","charge","coulomb","current","voltage","resistance",
  "ohm","capacitor","capacitance","inductor","magnetic field","flux","lorentz force",
  "emf","faraday","induction","ac","dc","wave","frequency","wavelength","optics",
  "reflection","refraction","lens","mirror","interference","diffraction",
  "photoelectric","photon","nuclear","radioactivity","half life","decay","quantum",
  // extra
  "pressure","density","viscosity","surface tension","temperature","heat","thermodynamics",
  "entropy","carnot","doppler","sound","light","mass","weight","normal force"
];

const chemistryKeywords = [
  "mole","molar","molarity","normality","equilibrium","constant","kc","kp",
  "reaction","rate","kinetics","order","activation energy","catalyst",
  "thermodynamics","enthalpy","entropy","gibbs","heat","temperature",
  "acid","base","ph","buffer","neutralization","titration",
  "electrochemistry","electrode","anode","cathode","cell","potential","nernst equation",
  "oxidation","reduction","redox","valency","bond","ionic","covalent",
  "hybridization","orbital","vsepr","structure","periodic","group","period",
  "organic","alkane","alkene","alkyne","benzene","aromatic","alcohol","phenol",
  "aldehyde","ketone","amine","polymer","isomer","stereochemistry",
  "nucleophile","electrophile","mechanism","substitution","elimination",
  // extra
  "atomic number","mass number","isotope","electron","proton","neutron",
  "concentration","stoichiometry","limiting reagent","yield","mol"
];

// ── Smart Subject Detector ─────────────────────────────────
// Returns: 'math' | 'physics' | 'chemistry' | 'other'
function detectSubject(text) {
  const q = text.toLowerCase();

  // Direct math symbol detection — these almost always mean math/science
  const hasMathSymbol = /[+\-*/=^%×÷√π∑∫{}]/.test(text) ||
    /\d+\s*[\+\-\*\/\=\^×÷]\s*\d+/.test(text) ||   // number operator number
    /[\(\)]\s*\d/.test(text) ||                        // bracket with number
    /\d+\s*[\(\)]/.test(text) ||                       // number with bracket
    /π|√|∑|∫|∞|θ|α|β|γ|λ|μ|σ|φ/.test(text);          // math symbols

  // Has number AND operator — real math, not "give me 4 examples"
  const hasNumberWithOperator = /\d/.test(q) && /[\+\-\*\/\=\^\%\×\÷\√]/.test(q);

  // Priority order: Math → Physics → Chemistry → Other
  if (mathKeywords.some(k => q.includes(k))) return 'math';
  if (physicsKeywords.some(k => q.includes(k))) return 'physics';
  if (chemistryKeywords.some(k => q.includes(k))) return 'chemistry';
  if (hasMathSymbol || hasNumberWithOperator) return 'math';

  return 'other';
}

function needsGemini(subject) {
  return subject === 'math' || subject === 'physics' || subject === 'chemistry';
}

// ── Gemini API Call ────────────────────────────────────────
async function callGemini(messages, apiKey) {
  // Convert OpenAI-style messages to Gemini format
  const systemMsg = messages.find(m => m.role === 'system');
  const userMessages = messages.filter(m => m.role !== 'system');

  const contents = userMessages.map(m => ({
    role: m.role === 'assistant' ? 'model' : 'user',
    parts: Array.isArray(m.content)
      ? m.content.map(c => c.type === 'text' ? { text: c.text } : { text: '' })
      : [{ text: m.content }]
  }));

  const body = {
    system_instruction: systemMsg ? { parts: [{ text: systemMsg.content }] } : undefined,
    contents,
    tools: [{ code_execution: {} }],
    generationConfig: {
      temperature: 0,
      maxOutputTokens: 1024,
    }
  };

  const resp = await fetch(
    `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=${apiKey}`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    }
  );

  const data = await resp.json();
  if (!resp.ok) throw new Error(data.error?.message || 'Gemini API error: ' + resp.status);

  // Extract text from Gemini response
  const candidate = data.candidates?.[0];
  if (!candidate) throw new Error('No candidates in Gemini response');

  const parts = candidate.content?.parts || [];
  let finalText = '';

  for (const part of parts) {
    if (part.text) finalText += part.text;
    if (part.codeExecutionResult?.output) {
      // Extract just the numeric result from code output and add as CALC line
      const output = part.codeExecutionResult.output.trim();
      const numMatch = output.match(/[\d\.]+/);
      if (numMatch) finalText += '\nCALC_RESULT: ' + numMatch[0];
    }
  }

  if (!finalText.trim()) throw new Error('Empty response from Gemini');

  // Return in OpenAI-compatible format
  return {
    choices: [{
      message: {
        role: 'assistant',
        content: finalText.trim()
      }
    }]
  };
}

// ── Groq API Call ──────────────────────────────────────────
async function callGroq(body, apiKey) {
  const resp = await fetch('https://api.groq.com/openai/v1/chat/completions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${apiKey}` },
    body: JSON.stringify({ ...body, model: 'llama-3.3-70b-versatile' })
  });
  return await resp.json();
}

// ── Main Handler ───────────────────────────────────────────
export default {
  async fetch(request, env) {
    if (request.method === 'OPTIONS') return new Response(null, { status: 204, headers: corsHeaders });
    if (request.method !== 'POST') return new Response('Method not allowed', { status: 405, headers: corsHeaders });

    try {
      const body = await request.json();

      const hasImage = body.messages && body.messages.some(m =>
        Array.isArray(m.content) && m.content.some(c => c.type === 'image_url')
      );

      // ── PHOTO: Step 1 — Scout reads image (always Groq) ──
      if (body._detectOnly && hasImage) {
        const lastUserMsg = body.messages[body.messages.length - 1];
        const allContent = Array.isArray(lastUserMsg.content) ? lastUserMsg.content : [];
        const imageItems = allContent.filter(c => c.type === 'image_url');

        const detectResp = await fetch('https://api.groq.com/openai/v1/chat/completions', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${env.GROQ_API_KEY}` },
          body: JSON.stringify({
            model: 'meta-llama/llama-4-scout-17b-16e-instruct',
            messages: [{ role: 'user', content: [...imageItems, { type: 'text', text: 'Read and transcribe ALL text visible in this image exactly as written.' }] }],
            max_tokens: 512, temperature: 0.1
          })
        });
        const detectData = await detectResp.json();
        return new Response(JSON.stringify(detectData), { status: detectResp.status, headers: { ...corsHeaders, 'Content-Type': 'application/json' } });
      }

      // ── PHOTO: Step 2 — Scout extracts, then smart route solves ──
      if (hasImage) {
        const lastUserMsg = body.messages[body.messages.length - 1];
        const allContent = Array.isArray(lastUserMsg.content) ? lastUserMsg.content : [];
        const imageItems = allContent.filter(c => c.type === 'image_url');
        const textItems = allContent.filter(c => c.type === 'text');

        // Step 1: Scout reads image
        const extractResp = await fetch('https://api.groq.com/openai/v1/chat/completions', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${env.GROQ_API_KEY}` },
          body: JSON.stringify({
            model: 'meta-llama/llama-4-scout-17b-16e-instruct',
            messages: [{ role: 'user', content: [...imageItems, { type: 'text', text: 'Read and write out ALL text, numbers, math expressions visible in the image.' }] }],
            max_tokens: 1024, temperature: 0.1
          })
        });
        const extractData = await extractResp.json();
        const extracted = extractData.choices?.[0]?.message?.content || '';

        // Step 2: Detect subject from extracted text
        const subject = detectSubject(extracted);
        const systemMsg = body.messages.find(m => m.role === 'system');
        const originalText = textItems.map(t => t.text).join(' ');
        const solveMessages = [];
        if (systemMsg) solveMessages.push(systemMsg);
        solveMessages.push({ role: 'user', content: `Question from image:\n\n${extracted}\n\n${originalText || 'Solve step by step.'}` });

        let result;
        if (needsGemini(subject)) {
          result = await callGemini(solveMessages, env.GEMINI_API_KEY);
        } else {
          result = await callGroq({ messages: solveMessages, max_tokens: body.max_tokens || 1024, temperature: 0.5 }, env.GROQ_API_KEY);
        }

        return new Response(JSON.stringify(result), { headers: { ...corsHeaders, 'Content-Type': 'application/json' } });
      }

      // ── TEXT QUESTION: Smart route ──
      const lastUserMsg = body.messages?.findLast(m => m.role === 'user');
      const questionText = typeof lastUserMsg?.content === 'string'
        ? lastUserMsg.content
        : lastUserMsg?.content?.map(c => c.text || '').join(' ') || '';

      const subject = detectSubject(questionText);
      let result;

      if (needsGemini(subject)) {
        result = await callGemini(body.messages, env.GEMINI_API_KEY);
      } else {
        result = await callGroq({ ...body, max_tokens: body.max_tokens || 1024 }, env.GROQ_API_KEY);
      }

      return new Response(JSON.stringify(result), { headers: { ...corsHeaders, 'Content-Type': 'application/json' } });

    } catch (err) {
      return new Response(JSON.stringify({ error: { message: err.message || 'Unknown error' } }), { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } });
    }
  }
};
