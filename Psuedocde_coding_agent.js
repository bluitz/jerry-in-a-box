const fs = require('fs');
const { execSync } = require('child_process');
const axios = require('axios'); // To call the MCP
const openai = require('openai');
openai.apiKey = 'sk-...';

const codeFile = './agent-workspace/index.js';
const testFile = './agent-workspace/test.js';
const goal = "Write a function `isPalindrome` and test it";

// Step 0: Ask the MCP server how to run tests
async function getTestCommandFromMCP() {
  const resp = await axios.get('http://localhost:3000/mcp/tvui/metadata');
  return resp.data.testCommand; // e.g., "npm test", "vitest run", or "node test.js"
}

const testCommand = await getTestCommandFromMCP();
console.log("📡 MCP returned test command:", testCommand);

// Step 1: Ask OpenAI to plan the steps
const planPrompt = `
You are an AI coding assistant. Your goal is: "${goal}"
Plan step-by-step what you would do, including writing and running unit tests.
Respond with a numbered list.
Example: "analyze → plan → code → test → repeat"
`;
const planResp = await openai.chat.completions.create({
  model: 'gpt-4',
  messages: [{ role: "user", content: planPrompt }]
});
const steps = parseSteps(planResp.choices[0].message.content);

// Step 2: Loop through the plan
for (let i = 0; i < steps.length; i++) {
  const step = steps[i];
  console.log(`🚧 Step ${i + 1}/${steps.length}: ${step}`);

  // Generate new code and test based on step
  const codeGenPrompt = `
You're working on step ${i + 1}: "${step}"
Here is the current index.js:

${fs.readFileSync(codeFile, 'utf-8')}

And here is test.js:

${fs.readFileSync(testFile, 'utf-8')}

Please update index.js and test.js based on this step.
`;
  const codeResp = await openai.chat.completions.create({
    model: 'gpt-4',
    messages: [{ role: "user", content: codeGenPrompt }]
  });

  const { indexJs, testJs } = extractFiles(codeResp.choices[0].message.content);
  fs.writeFileSync(codeFile, indexJs);
  fs.writeFileSync(testFile, testJs);

  // Step 3: Run tests using MCP-discovered command
  try {
    execSync(testCommand, { stdio: 'inherit' });
    console.log(`✅ Step ${i + 1} passed tests`);
  } catch (err) {
    console.log(`❌ Tests failed. Retrying or re-prompting...`);
    // (optional: extract error logs and feed back to GPT for retry)
  }
}
