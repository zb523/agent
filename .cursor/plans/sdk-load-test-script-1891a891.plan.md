<!-- 1891a891-1d6c-48f2-8592-0ff57c4837ed 235a26b4-efd4-4f15-a924-f36c46628346 -->
# Fix Load Test - Use Native SDK

## Problem

The current script uses `livekit-client` which is browser-only and throws:

```
LiveKit doesn't seem to be supported on this browser
```

## Solution

Switch to `@livekit/rtc-node` — LiveKit's native Node.js SDK with built-in WebRTC bindings.

---

## Files to Edit

### File 1: `/Users/zubeyr/Documents/GitHub/listener-website/package.json`

Add two new devDependencies:

- `@livekit/rtc-node` (native WebRTC SDK)
- `livekit-server-sdk` (for local token generation)

---

### File 2: `/Users/zubeyr/Documents/GitHub/listener-website/scripts/load-test.ts`

Complete rewrite:

- Import from `@livekit/rtc-node` instead of `livekit-client`
- Use `livekit-server-sdk` AccessToken for local token generation (no API calls)
- Use `room.registerTextStreamHandler()` for text streams
- Keep all diagnostics logic (timestamps, percentiles, summary)

---

### File 3: `/Users/zubeyr/Documents/GitHub/listener-website/.env.example`

Add LiveKit credentials:

```
WORKER_URL=https://baian-server.zubeyrbarre.workers.dev
LIVEKIT_API_KEY=your_api_key
LIVEKIT_API_SECRET=your_api_secret
LIVEKIT_URL=wss://your-project.livekit.cloud
```

---

## Usage After Fix

```bash
cd listener-website
npm install
# Edit .env with your actual LiveKit creds
npm run load-test -- --slug zubeyr-slug --listeners 50
```

### To-dos

- [ ] Add @livekit/rtc-node and livekit-server-sdk to package.json
- [ ] Rewrite load-test.ts with native SDK imports and local token generation
- [ ] Update .env.example with LIVEKIT_API_KEY, LIVEKIT_API_SECRET, LIVEKIT_URL