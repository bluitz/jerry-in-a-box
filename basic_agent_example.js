// npm install openai node-fetch dotenv
const { Configuration, OpenAIApi } = require("openai");
require('dotenv').config();

const configuration = new Configuration({
  apiKey: process.env.OPENAI_API_KEY,
});
const openai = new OpenAIApi(configuration);

// --- Tools the agent can use ---
const tools = {
  search: (query) => `🔍 Searching for "${query}"...`,
  calculator: (expr) => `🧮 Result: ${eval(expr)}`,
};

// --- Initial agent state ---
let step = 0;
let goal = "Find the capital of France and multiply the length of its name by 3";

// --- Main agent loop ---
async function runAgent() {
  while (true) {
    console.log(`\n🌀 Step ${step + 1}`);

    // Send current state to OpenAI
    const prompt = `
You are an AI agent. Your goal is: ${goal}
Decide which tool to use: "search" or "calculator", and what input to give it.
Respond in JSON like:
{ "tool": "search", "input": "..." }

Current step: ${step}`;

    const completion = await openai.createChatCompletion({
      model: "gpt-4o",
      messages: [{ role: "user", content: prompt }],
    });

    const reply = completion.data.choices[0].message.content;
    console.log("🤖 OpenAI says:", reply);

    let action = JSON.parse(reply);

    // Call the chosen tool
    const output = tools[action.tool]?.(action.input);

    console.log("🛠️ Tool output:", output);
    step++;
  }
}

runAgent();
