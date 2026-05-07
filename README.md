# UAE_Green Advisor

🌿 Plant Care & Landscaping Advisor Agent

****Phase 1:** Foundation & System Design**

**Goal:** Define the agent's brain before writing a single line of code.
**Tasks:**
- Define the agent's persona (friendly UAE plant expert, bilingual EN/AR)
- Write the master system prompt with UAE-specific knowledge baked in (extreme heat, hard water, desert soil, common UAE plants like palms, bougainvillea, desert rose, ficus)
- List the 20 most common UAE plant problems and their solutions as the agent's knowledge base
- Define input types the agent will accept (text description, image, or both)
- Map out the conversation flow (greeting → diagnosis → recommendation → follow-up)

**Phase 2: Core Diagnosis Engine**

**Goal:** Build the single-turn diagnosis — user describes/shows a problem, agent responds with a diagnosis.
**Tasks:**
- Set up the Groq API call with the system prompt from Phase 1
- Build the structured response format (plant identified → problem diagnosed → severity level → treatment steps)
- Test with 10 real UAE plant scenarios (text-only and image-based)
- Refine the system prompt based on test results
- Add Arabic language support (detect language and respond accordingly)

**Phase 3: UAE Weather & Context Integration**

**Goal:** Make the agent aware of real-time UAE conditions so advice is seasonally accurate.
**Tasks:**
- Connect OpenWeatherMap free API for Dubai, Abu Dhabi, and Sharjah
- Inject live weather data (temperature, humidity, UV index) into every agent prompt automatically
- Add seasonal logic (summer heat stress warnings, post-sandstorm care tips, winter planting window advice)
- Add a city selector so users specify their emirate
- Test how advice changes across seasons and locations

**Phase 4: Frontend UI Build**

**Goal:** Build a clean, mobile-friendly interface that feels professional.
**Tasks:**
- Design the chat UI with plant-themed aesthetics (greens, earthy tones)
- Build image upload component with preview
- Add city/emirate selector dropdown
- Build the chat message display (user vs agent bubbles)
- Add a typing indicator while agent is thinking
- Make it fully responsive for mobile (most UAE users are on phones)
- Add bilingual toggle (English / Arabic with RTL support)
