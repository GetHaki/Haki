/** Minimal Haki agent loop: context -> LLM -> capture_turn.
 *
 * Run: HAKI_API_KEY=hk_... node examples/basic-agent.mjs
 * (expects a Haki API on HAKI_API_URL, default http://localhost:8100)
 */
import { HakiClient, buildPromptContext, captureTurn } from "../dist/index.js";

const client = new HakiClient({
  baseUrl: process.env.HAKI_API_URL ?? "http://localhost:8100",
  apiKey: process.env.HAKI_API_KEY,
});
const subject = {
  subjectId: process.env.HAKI_SUBJECT_ID ?? "usr_42",
  projectId: process.env.HAKI_PROJECT_ID ?? "prj_haki_verify",
};
const userMsg = process.argv[2] ?? "dans quelle langue envoyer les factures ?";

const { packet } = await client.context({ ...subject, query: userMsg });
const prompt = buildPromptContext(packet) + "\nYou are a helpful support agent.";
const answer = `prompt ready (${packet.facts.length} memory fact(s) injected)`; // ← your LLM call here
console.log(prompt);
await captureTurn(client, { ...subject, userMsg, assistantMsg: answer });
console.log("turn captured — the exchange is now part of Haki memory");
