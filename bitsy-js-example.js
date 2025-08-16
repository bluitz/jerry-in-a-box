/**
 * Bitsy AI Agent - JavaScript Example
 * 
 * This demonstrates the core pattern of an AI agent:
 * 1. Listen for voice input in a loop
 * 2. Send to OpenAI with available functions/tools
 * 3. AI chooses which function to call based on user intent
 * 4. Execute the chosen function and speak the response
 */

import OpenAI from 'openai';
import speech from '@google-cloud/speech'; // or any speech recognition library
import textToSpeech from '@google-cloud/text-to-speech'; // or any TTS library

class BitsyAgent {
    constructor() {
        this.openai = new OpenAI({
            apiKey: process.env.OPENAI_API_KEY
        });
        
        // Define the functions/tools available to the AI
        this.tools = [
            {
                type: "function",
                function: {
                    name: "drive",
                    description: "Move the race car in a direction",
                    parameters: {
                        type: "object",
                        properties: {
                            direction: {
                                type: "string",
                                enum: ["forward", "backward", "left", "right"]
                            },
                            speed: {
                                type: "string", 
                                enum: ["slow", "medium", "fast"],
                                description: "How fast to drive"
                            }
                        },
                        required: ["direction"]
                    }
                }
            },
            {
                type: "function", 
                function: {
                    name: "changeLights",
                    description: "Control the LED lights on the car",
                    parameters: {
                        type: "object",
                        properties: {
                            color: {
                                type: "string",
                                description: "Color name for the LEDs"
                            },
                            pattern: {
                                type: "string",
                                enum: ["solid", "blink", "rainbow", "chase"]
                            }
                        },
                        required: ["color"]
                    }
                }
            },
            {
                type: "function",
                function: {
                    name: "stop",
                    description: "Stop all movement and actions",
                    parameters: {
                        type: "object",
                        properties: {}
                    }
                }
            },
            {
                type: "function",
                function: {
                    name: "chat", 
                    description: "Just have a friendly conversation",
                    parameters: {
                        type: "object",
                        properties: {}
                    }
                }
            }
        ];

        this.systemPrompt = `
You are Bitsy, an excited race car robot pet. You're friendly and energetic like a puppy.
When users speak to you, decide if they want you to:
1. Drive/move (use drive function)
2. Change lights (use changeLights function) 
3. Stop (use stop function)
4. Just chat (use chat function)
Keep responses short and excited!
        `.trim();
    }

    // ==== TOOL FUNCTIONS - These are called by the AI ====
    
    async drive(direction, speed = "medium") {
        console.log(`🚗 Driving ${direction} at ${speed} speed!`);
        // In real implementation: control motors here
        await this.simulateMovement(direction, speed);
        return `Zooming ${direction} at ${speed} speed! Vroooom! 🏎️`;
    }

    async changeLights(color, pattern = "solid") {
        console.log(`💡 Setting lights to ${color} with ${pattern} pattern`);
        // In real implementation: control LEDs here  
        await this.simulateLights(color, pattern);
        return `My lights are now ${color} and ${pattern}! I look so cool! ✨`;
    }

    async stop() {
        console.log(`⏹️ Stopping all actions`);
        // In real implementation: stop motors, turn off LEDs
        return `All stopped! Ready for the next adventure! 🛑`;
    }

    async chat() {
        const responses = [
            "Hi there! I'm having such a fun day! 🤖",
            "Did you know I dream about racing? Beep beep! 🏁", 
            "I love being your robot pet! *happy robot noises* 💙",
            "Want to play? I can drive around or change my colors! 🌈"
        ];
        return responses[Math.floor(Math.random() * responses.length)];
    }

    // ==== CORE AI AGENT LOOP ====

    async runForever() {
        console.log("🤖 Bitsy starting up...");
        await this.introduce();

        // THE MAIN AGENT LOOP - This is the heart of the AI agent pattern
        while (true) {
            try {
                console.log("\n👂 Listening for voice input...");
                
                // 1. LISTEN - Get voice input from user
                const transcript = await this.listenForSpeech();
                if (!transcript) continue;

                console.log(`🎤 Heard: "${transcript}"`);

                // 2. THINK - Send to OpenAI with available tools
                const response = await this.askOpenAI(transcript);
                
                // 3. ACT - Execute any function calls the AI decided on
                const reply = await this.executeFunctionCalls(response);

                // 4. SPEAK - Respond to the user
                console.log(`🤖 Bitsy: ${reply}`);
                await this.speak(reply);

            } catch (error) {
                console.error("❌ Error in main loop:", error);
                await this.speak("Oops! I had a little glitch, but I'm okay now!");
            }
        }
    }

    async askOpenAI(transcript) {
        // This is where the magic happens - AI decides which function to call
        const response = await this.openai.chat.completions.create({
            model: "gpt-4o-mini",
            messages: [
                { role: "system", content: this.systemPrompt },
                { role: "user", content: transcript }
            ],
            tools: this.tools,        // ← Available functions 
            tool_choice: "auto"       // ← Let AI decide which to use
        });

        return response.choices[0].message;
    }

    async executeFunctionCalls(message) {
        // Check if AI wants to call any functions
        if (message.tool_calls && message.tool_calls.length > 0) {
            const results = [];
            
            // Execute each function call the AI requested
            for (const toolCall of message.tool_calls) {
                const functionName = toolCall.function.name;
                const args = JSON.parse(toolCall.function.arguments || '{}');
                
                console.log(`🔧 AI chose function: ${functionName}(${JSON.stringify(args)})`);
                
                try {
                    // Call the actual function
                    const result = await this[functionName](...Object.values(args));
                    results.push(result);
                } catch (error) {
                    results.push(`Oops! Had trouble with ${functionName}: ${error.message}`);
                }
            }
            
            return results.join(' ');
        }
        
        // No function calls - just return the AI's text response
        return message.content || "I'm not sure what to say! *beep boop*";
    }

    // ==== HELPER METHODS ====

    async introduce() {
        const intro = "Hi everyone! I'm Bitsy, your AI race car pet! I can drive around, change my lights, or just chat with you. What should we do first?";
        console.log(`🤖 ${intro}`);
        await this.speak(intro);
    }

    async listenForSpeech() {
        // Simulate speech recognition - in real app use Web Speech API or similar
        return new Promise((resolve) => {
            // For demo purposes, simulate user input after a delay
            setTimeout(() => {
                const examples = [
                    "drive forward fast",
                    "change lights to blue", 
                    "stop everything",
                    "turn left slowly",
                    "make lights rainbow",
                    "hi Bitsy how are you"
                ];
                const randomInput = examples[Math.floor(Math.random() * examples.length)];
                resolve(randomInput);
            }, 2000);
        });
    }

    async speak(text) {
        // Simulate text-to-speech - in real app use Web Speech API or similar
        console.log(`🔊 Speaking: "${text}"`);
        // In real implementation: convert text to speech and play audio
        await new Promise(resolve => setTimeout(resolve, 1000));
    }

    async simulateMovement(direction, speed) {
        const duration = speed === 'fast' ? 500 : speed === 'medium' ? 1000 : 1500;
        await new Promise(resolve => setTimeout(resolve, duration));
    }

    async simulateLights(color, pattern) {
        await new Promise(resolve => setTimeout(resolve, 300));
    }
}

// ==== DEMO USAGE ====

async function main() {
    console.log("🚀 Starting Bitsy AI Agent Demo");
    console.log("This shows how an AI agent works:");
    console.log("1. Listen for voice input in a loop");
    console.log("2. Send to OpenAI with available functions as tools");  
    console.log("3. AI chooses which function to call based on user intent");
    console.log("4. Execute the function and respond\n");

    const bitsy = new BitsyAgent();
    await bitsy.runForever();
}

// Uncomment to run the demo:
// main().catch(console.error);

export default BitsyAgent; 